"""Observe native context peer mappings and reset-verified D2D/Graph copies.

Run through a finite native admission guard. Values and transfer routes are not
simulator assertions; a successful API call still requires independent readback.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path


def main():
    admission = json.loads(Path(os.environ['NATIVE_ADMISSION']).read_text())
    lock = os.fstat(int(os.environ['NATIVE_LOCK_FD']))
    if (lock.st_dev, lock.st_ino) != (admission['lock_device'], admission['lock_inode']):
        raise ValueError('Admission lock identity changed')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != admission['boot_id']:
        raise ValueError('Admission boot identity changed')
    library = admission['native_library']
    if hashlib.sha256(Path(library['path']).read_bytes()).hexdigest() != library['sha256']:
        raise ValueError('Native Driver changed')
    if os.getppid() != admission['parent_pid']:
        raise ValueError('Admission parent changed')
    driver = C.CDLL(library['path'])
    P, U, I, S = C.c_void_p, C.c_uint64, C.c_int, C.c_size_t

    def bind(name, args):
        fn = getattr(driver, name)
        fn.argtypes, fn.restype = args, I
        return fn

    init = bind('cuInit', [C.c_uint])
    count = bind('cuDeviceGetCount', [C.POINTER(I)])
    create = bind('cuCtxCreate_v2', [C.POINTER(P), C.c_uint, I])
    destroy = bind('cuCtxDestroy_v2', [P])
    set_current = bind('cuCtxSetCurrent', [P])
    can_access = bind('cuDeviceCanAccessPeer', [C.POINTER(I), I, I])
    get_uuid = bind('cuDeviceGetUuid_v2', [P, I])
    get_pci = bind('cuDeviceGetPCIBusId', [P, I, I])
    enable = bind('cuCtxEnablePeerAccess', [P, C.c_uint])
    disable = bind('cuCtxDisablePeerAccess', [P])
    alloc = bind('cuMemAlloc_v2', [C.POINTER(U), S])
    free = bind('cuMemFree_v2', [U])
    h2d = bind('cuMemcpyHtoD_v2', [U, P, S])
    d2h = bind('cuMemcpyDtoH_v2', [P, U, S])
    copy = bind('cuMemcpyDtoDAsync_v2', [U, U, S, P])
    stream_create = bind('cuStreamCreate', [C.POINTER(P), C.c_uint])
    stream_sync = bind('cuStreamSynchronize', [P])
    stream_destroy = bind('cuStreamDestroy_v2', [P])
    begin = bind('cuStreamBeginCapture_v2', [P, I])
    end = bind('cuStreamEndCapture', [P, C.POINTER(P)])
    instantiate = bind('cuGraphInstantiateWithFlags', [C.POINTER(P), P, C.c_uint64])
    graph_destroy = bind('cuGraphDestroy', [P])
    exec_destroy = bind('cuGraphExecDestroy', [P])
    launch = bind('cuGraphLaunch', [P, P])
    attribute = bind('cuPointerGetAttribute', [P, I, U])

    def ok(result):
        if result:
            raise RuntimeError('Native Driver API error ' + str(result))

    ok(init(0))
    devices = I()
    ok(count(C.byref(devices)))
    if devices.value != 2:
        raise ValueError('This diagnostic requires exactly two admitted devices')
    contexts, streams, pointers = [P(), P()], [P(), P()], [U(), U()]
    values = (C.c_uint32 * 4096)(*[0x13579 + i * 13 for i in range(4096)])
    zeros, output = (C.c_uint32 * 4096)(), (C.c_uint32 * 4096)()
    size = C.sizeof(values)
    report = {'scope': 'native pointers and mapped D2D; copy route unmeasured',
              'native_library': library, 'devices': [], 'directions': [], 'cases': []}
    for device in range(2):
        raw_uuid, pci = (C.c_ubyte * 16)(), C.create_string_buffer(32)
        ok(get_uuid(raw_uuid, device))
        ok(get_pci(pci, len(pci), device))
        hex_uuid = bytes(raw_uuid).hex()
        uuid = 'GPU-' + '-'.join([hex_uuid[:8], hex_uuid[8:12], hex_uuid[12:16], hex_uuid[16:20], hex_uuid[20:]])
        if uuid not in admission['assigned_gpu_uuids']:
            raise ValueError('Driver ordinal UUID is outside native admission')
        report['devices'].append({'ordinal': device, 'uuid': uuid, 'pci': pci.value.decode()})
        ok(create(C.byref(contexts[device]), 0, device))
        ok(alloc(C.byref(pointers[device]), size))
        ok(stream_create(C.byref(streams[device]), 1))

    attributes = [(1, 'context', P), (2, 'memory_type', C.c_uint),
                  (3, 'device_pointer', U), (9, 'device_ordinal', I),
                  (11, 'range_start', U), (12, 'range_size', S), (13, 'mapped', C.c_uint)]

    def snapshot(observer, owner):
        ok(set_current(contexts[observer]))
        rows = []
        for code, name, dtype in attributes:
            value = dtype()
            result = attribute(C.byref(value), code, pointers[owner])
            rows.append({'attribute': name, 'exit': result, 'value': value.value})
        return rows

    def verify_snapshot(rows, observer, owner, enabled):
        for row in rows:
            expected_exit = 0 if row['attribute'] != 'device_pointer' or enabled else 1
            if row['exit'] != expected_exit:
                raise ValueError('Unexpected pointer attribute result: ' + str(row))
        expected = {'context': contexts[owner].value, 'memory_type': 2, 'device_ordinal': owner,
                    'range_start': pointers[owner].value, 'range_size': size, 'mapped': 1}
        if enabled:
            expected['device_pointer'] = pointers[owner].value
        if any(row['value'] != expected[row['attribute']] for row in rows if row['attribute'] in expected):
            raise ValueError('Unexpected pointer ownership or mapping value')

    for observer, owner in [(0, 1), (1, 0)]:
        ok(set_current(contexts[observer]))
        can = I()
        ok(can_access(C.byref(can), observer, owner))
        direction = {'observer': observer, 'owner': owner, 'can_access': can.value,
                     'before': snapshot(observer, owner)}
        direction['enable_exit'] = enable(contexts[owner], 0)
        direction['enabled'] = snapshot(observer, owner)
        if direction['enable_exit'] != (0 if can.value else 217):
            raise ValueError('Unexpected peer enable result')
        verify_snapshot(direction['before'], observer, owner, False)
        verify_snapshot(direction['enabled'], observer, owner, bool(can.value))
        if can.value and not direction['enable_exit']:
            for mode in ['eager', 'graph']:
                for use in ['read_peer', 'write_peer']:
                    src, dst = (owner, observer) if use == 'read_peer' else (observer, owner)
                    graph, executable = P(), P()
                    if mode == 'graph':
                        ok(set_current(contexts[observer]))
                        ok(begin(streams[observer], 0))
                        ok(copy(pointers[dst], pointers[src], size, streams[observer]))
                        ok(end(streams[observer], C.byref(graph)))
                        ok(instantiate(C.byref(executable), graph, 0))
                        ok(graph_destroy(graph))
                    for replay in range(2):
                        values[0] = 0x13579 + observer * 4096 + replay * 257 + (31 if use == 'write_peer' else 0)
                        ok(set_current(contexts[src]))
                        ok(h2d(pointers[src], values, size))
                        ok(d2h(output, pointers[src], size))
                        source_mismatches = sum(x != y for x, y in zip(output, values))
                        if source_mismatches:
                            raise ValueError('Source upload readback failed')
                        ok(set_current(contexts[dst]))
                        ok(h2d(pointers[dst], zeros, size))
                        ok(d2h(output, pointers[dst], size))
                        reset_mismatches = sum(x != 0 for x in output)
                        if reset_mismatches:
                            raise ValueError('Destination reset failed')
                        ok(set_current(contexts[observer]))
                        if mode == 'graph':
                            ok(launch(executable, streams[observer]))
                        else:
                            ok(copy(pointers[dst], pointers[src], size, streams[observer]))
                        ok(stream_sync(streams[observer]))
                        ok(set_current(contexts[dst]))
                        ok(d2h(output, pointers[dst], size))
                        mismatches = sum(x != y for x, y in zip(output, values))
                        report['cases'].append({'observer': observer, 'owner': owner, 'source': src,
                            'destination': dst, 'mode': mode, 'use': use, 'replay': replay,
                            'source_mismatches': source_mismatches,
                            'reset_mismatches': reset_mismatches, 'mismatches': mismatches,
                            'expected_sha256': hashlib.sha256(bytes(values)).hexdigest(),
                            'observed_sha256': hashlib.sha256(bytes(output)).hexdigest(),
                            'count': len(values), 'status': 'FAIL' if mismatches else 'PASS'})
                        ok(set_current(contexts[observer]))
                    if executable.value:
                        ok(exec_destroy(executable))
        else:
            for mode in ['eager', 'graph']:
                for use in ['read_peer', 'write_peer']:
                    for replay in range(2):
                        report['cases'].append({'observer': observer, 'owner': owner,
                            'mode': mode, 'use': use, 'replay': replay,
                            'status': 'SKIP_UNSUPPORTED', 'reason': 'Native cuDeviceCanAccessPeer=0'})
        ok(set_current(contexts[observer]))
        direction['disable_exit'] = disable(contexts[owner])
        direction['disabled'] = snapshot(observer, owner)
        if direction['disable_exit'] != (0 if can.value else 705):
            raise ValueError('Unexpected peer disable result')
        verify_snapshot(direction['disabled'], observer, owner, False)
        report['directions'].append(direction)
    for device in [1, 0]:
        ok(set_current(contexts[device]))
        ok(free(pointers[device]))
        ok(stream_destroy(streams[device]))
        ok(destroy(contexts[device]))
    report['counts'] = {name: sum(row['status'] == name for row in report['cases'])
                       for name in ['PASS', 'FAIL', 'SKIP_UNSUPPORTED']}
    report['pointer_attribute_checks'] = 42
    print(json.dumps(report, indent=2), flush=True)
    return int(bool(report['counts']['FAIL']))


if __name__ == '__main__':
    raise SystemExit(main())
