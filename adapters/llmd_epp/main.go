// Embed the pinned llm-d Router profile; policies remain in upstream packages.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"

	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fw "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/queuedepth"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type endpoint struct {
	ID      string `json:"id"`
	Queue   int    `json:"queue"`
	Enabled bool   `json:"enabled"`
}
type decision struct {
	Picked string             `json:"picked"`
	Scores map[string]float64 `json:"scores"`
}

func main() {
	filter, err := bylabel.NewSelector("enabled", &metav1.LabelSelector{MatchLabels: map[string]string{"enabled": "true"}})
	if err != nil {
		panic(err)
	}
	profile := scheduling.NewSchedulerProfile().WithFilters(filter).
		WithScorers(scheduling.NewWeightedScorer(queuedepth.NewQueueScorer(), 1)).
		WithPicker(maxscore.NewMaxScorePicker(1))
	scanner := bufio.NewScanner(os.Stdin)
	encoder := json.NewEncoder(os.Stdout)
	for scanner.Scan() {
		var input []endpoint
		if err := json.Unmarshal(scanner.Bytes(), &input); err != nil {
			panic(err)
		}
		candidates := make([]fw.Endpoint, 0, len(input))
		for _, e := range input {
			if e.ID == "" || e.Queue < 0 {
				panic("invalid endpoint snapshot")
			}
			candidates = append(candidates, fw.NewEndpoint(&dl.EndpointMetadata{Name: e.ID, ID: dl.ID{Name: e.ID, Namespace: "fakecuda"},
				Labels: map[string]string{"enabled": fmt.Sprint(e.Enabled)}}, &dl.Metrics{WaitingQueueSize: e.Queue}, nil))
		}
		result, err := profile.Run(context.Background(), &fw.InferenceRequest{}, candidates)
		if err != nil {
			panic(err)
		}
		output := decision{Picked: result.TargetEndpoints[0].GetMetadata().Name, Scores: map[string]float64{}}
		for _, scored := range result.ScoredCandidates {
			output.Scores[scored.GetMetadata().Name] = scored.Score
		}
		if err := encoder.Encode(output); err != nil {
			panic(err)
		}
	}
	if err := scanner.Err(); err != nil {
		panic(err)
	}
}
