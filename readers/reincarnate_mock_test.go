// Copyright (c) Mainflux
// SPDX-License-Identifier: Apache-2.0

package readers

import (
	"fmt"
	"testing"
)

func TestARXMCAPredict(t *testing.T) {
	data := makeARXTestData(20)
	pred, err := ARXMCAPredict(data)
	if err != nil {
		t.Fatalf("ARXMCAPredict: %v", err)
	}
	if pred == 0 {
		t.Error("expected non-zero prediction")
	}
}

func TestARXMCAPredict_TooShort(t *testing.T) {
	data := makeARXTestData(10)
	_, err := ARXMCAPredict(data)
	if err == nil {
		t.Error("expected error for short dataset")
	}
}

func TestARXMCAPredictOLS(t *testing.T) {
	data := makeARXTestData(100)
	res, err := ARXMCAPredictOLS(data)
	if err != nil {
		t.Fatalf("ARXMCAPredictOLS: %v", err)
	}
	if len(res.Coefficients) != 4 {
		t.Errorf("expected 4 coefficients, got %d", len(res.Coefficients))
	}
	if len(res.Predictions) != len(res.Residuals) {
		t.Error("predictions and residuals length mismatch")
	}
	if res.LastPredicted == 0 && len(res.Predictions) > 0 {
		t.Error("expected non-zero last prediction")
	}
}

func TestARXMCAPredictOLS_Empty(t *testing.T) {
	_, err := ARXMCAPredictOLS(nil)
	if err == nil {
		t.Error("expected error for empty dataset")
	}
}

func makeARXTestData(n int) []ARXDataPoint {
	data := make([]ARXDataPoint, n)
	for i := 0; i < n; i++ {
		day := 15 + (i / 24)
		hour := i % 24
		// Add variation to avoid singular design matrix in OLS
		rh := 50.0 + float64(i%40) + float64((i*7)%11)*0.3
		temp := 10.0 + float64(i%15) + float64((i*3)%7)*0.5
		mca := 10.0 + float64(i%8)*0.7 + float64((i*5)%13)*0.1
		data[i] = ARXDataPoint{
			Time:      fmt.Sprintf("2024-04-%02dT%02d:00:00", day, hour),
			AmbientRH: rh,
			AmbientT:  temp,
			SensorMCA: mca,
			SensorMCB: mca + 0.5,
		}
	}
	return data
}
