package sim

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadEDPPCoeffs_FrozenLlama70b(t *testing.T) {
	path := filepath.Join("..", "scripts", "calibration", "coeffs-llama70b-h100-tp4.json")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("frozen coeffs not present: %v", err)
	}
	c, err := LoadEDPPCoeffs(path)
	if err != nil {
		t.Fatalf("LoadEDPPCoeffs: %v", err)
	}
	cases := []struct {
		name string
		got  float64
		want float64
	}{
		{"AlphaD", c.AlphaD, 16613.539607002218},
		{"AlphaP", c.AlphaP, 16617.95001666865},
		{"C0", c.C0, 5.347233255096054},
		{"C1", c.C1, 0.0476140288153138},
		{"CPf", c.CPf, 6.1446533788622295},
		{"CAttn", c.CAttn, 0.00010075607622433406},
	}
	for _, tc := range cases {
		if math.Abs(tc.got-tc.want) > 1e-9 {
			t.Errorf("%s = %v, want %v", tc.name, tc.got, tc.want)
		}
	}
}

func TestEDPPCoeffs_Validate(t *testing.T) {
	good := EDPPCoeffs{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: 0}
	if err := good.validate(); err != nil {
		t.Fatalf("good coeffs rejected: %v", err)
	}
	bad := []EDPPCoeffs{
		{AlphaD: 0, AlphaP: 1000, C0: 100, C1: 1, CPf: 10},      // AlphaD must be > 0
		{AlphaD: 1000, AlphaP: 1000, C0: -1, C1: 1, CPf: 10},    // C0 must be >= 0
		{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: -1}, // CAttn must be >= 0
		{AlphaD: 1000, AlphaP: 2000, C0: 100, C1: 1, CPf: 10},   // AlphaD/AlphaP diverge > 10%
	}
	for i, c := range bad {
		if err := c.validate(); err == nil {
			t.Errorf("case %d: expected validation error, got nil", i)
		}
	}
}
