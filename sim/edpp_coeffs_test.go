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

func TestEDPPCoeffs_Wp(t *testing.T) {
	c := EDPPCoeffs{CPf: 10, CAttn: 0}
	// linear-only (test model): W_p(300) = 10·300 = 3000
	if got := c.Wp(300); math.Abs(got-3000) > 1e-9 {
		t.Errorf("Wp(300) = %v, want 3000", got)
	}
	// with attention curvature: W_p(100) = 6·100 + (0.5/2)·100² = 600 + 0.25·10000 = 3100
	c2 := EDPPCoeffs{CPf: 6, CAttn: 0.5}
	if got := c2.Wp(100); math.Abs(got-3100) > 1e-9 {
		t.Errorf("Wp(100) = %v, want 3100", got)
	}
}

func TestEDPPCoeffs_IterTimeAndMu(t *testing.T) {
	c := EDPPCoeffs{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10}
	// T_iter decode at B=2, KV=2048, S_pf=0: 1000 + 100·2 + 1·2048 = 3248
	if got := c.tIterDecode(2, 2048, 0); math.Abs(got-3248) > 1e-9 {
		t.Errorf("tIterDecode = %v, want 3248", got)
	}
	// μ_dec = 1 − 1000/3248
	if got := c.muDecode(2, 2048, 0); math.Abs(got-(1-1000.0/3248)) > 1e-9 {
		t.Errorf("muDecode = %v, want %v", got, 1-1000.0/3248)
	}
	// T_iter prefill at S_pf=512: 1000 + 10·512 = 6120; μ_pf = 1 − 1000/6120
	if got := c.muPrefill(512); math.Abs(got-(1-1000.0/6120)) > 1e-9 {
		t.Errorf("muPrefill = %v, want %v", got, 1-1000.0/6120)
	}
	// deltaBarDecode(2048) = 100 + 1·2048 = 2148
	if got := c.deltaBarDecode(2048); math.Abs(got-2148) > 1e-9 {
		t.Errorf("deltaBarDecode = %v, want 2148", got)
	}
}

func TestEDPPCoeffs_MuClamped(t *testing.T) {
	// Degenerate α ≥ T_iter ⇒ μ floored at edppMinMu, never <= 0.
	c := EDPPCoeffs{AlphaD: 10_000, AlphaP: 10_000, C0: 0, C1: 0, CPf: 1}
	if got := c.muDecode(0, 0, 0); got < edppMinMu-1e-12 || got > 1.0 {
		t.Errorf("muDecode clamp = %v, want in [%v,1]", got, edppMinMu)
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
