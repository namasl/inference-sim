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
	// Linear-only (no attention), no cache: W_p(300,300) = 10·300 = 3000.
	if got := c.Wp(300, 300); math.Abs(got-3000) > 1e-9 {
		t.Errorf("Wp(300,300) = %v, want 3000", got)
	}
	// Causal basis, no cache: W_p(100,100) = 6·100 + 0.5·100·(100−50) = 600 + 2500 = 3100.
	c2 := EDPPCoeffs{CPf: 6, CAttn: 0.5}
	if got := c2.Wp(100, 100); math.Abs(got-3100) > 1e-9 {
		t.Errorf("Wp(100,100) = %v, want 3100", got)
	}
}

func TestWp_CausalBasis(t *testing.T) {
	c := EDPPCoeffs{CPf: 6.0, CAttn: 0.001}
	// No cache (a_p = a_r = 1000): C_pf·1000 + C_attn·1000·(1000 − 500) = 6000 + 500.
	got := c.Wp(1000, 1000)
	want := 6.0*1000 + 0.001*1000*(1000-1000.0/2)
	if got != want {
		t.Fatalf("Wp(1000,1000) = %v, want %v", got, want)
	}
	// Causal signature: at no cache the attention term equals (C_attn/2)·a_r².
	if attn := got - 6.0*1000; math.Abs(attn-(0.001/2)*1000*1000) > 1e-9 {
		t.Fatalf("no-cache causal attention = %v, want (C_attn/2)a² = %v", attn, (0.001/2)*1000*1000)
	}
	// Cached prefix (a_p=200 uncached of a_r=1000): C_pf·200 + C_attn·200·(1000 − 100).
	gotc := c.Wp(200, 1000)
	wantc := 6.0*200 + 0.001*200*(1000-200.0/2)
	if gotc != wantc {
		t.Fatalf("Wp(200,1000) = %v, want %v", gotc, wantc)
	}
}

func TestWd_DiscreteDecodeSum(t *testing.T) {
	c := EDPPCoeffs{C0: 5.0, C1: 0.05}
	// Wd = Σ_{k=0}^{o-1}(C0 + C1·(a_r+k)) = C0·o + C1·o·(a_r + (o-1)/2).
	sum := func(ar int, o int) float64 {
		var s float64
		for k := 0; k < o; k++ {
			s += 5.0 + 0.05*float64(ar+k)
		}
		return s
	}
	for _, tc := range []struct{ ar, o int }{{1000, 0}, {1000, 1}, {500, 10}, {2000, 128}} {
		got := c.Wd(tc.ar, float64(tc.o))
		want := sum(tc.ar, tc.o)
		if diff := got - want; diff < -1e-6 || diff > 1e-6 {
			t.Fatalf("Wd(%d,%d) = %v, want %v (discrete sum)", tc.ar, tc.o, got, want)
		}
	}
	if c.Wd(1000, 0) != 0 {
		t.Fatalf("Wd with o=0 must be 0, got %v", c.Wd(1000, 0))
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
	// Symmetric clamp for prefill: AlphaP=10_000, CPf=1 ⇒ tIterPrefill(0)=10_000,
	// 1 - AlphaP/tIterPrefill = 0 ⇒ floored at edppMinMu.
	cPf := EDPPCoeffs{AlphaP: 10_000, CPf: 1}
	if got := cPf.muPrefill(0); got < edppMinMu-1e-12 || got > 1.0 {
		t.Errorf("muPrefill clamp = %v, want in [%v,1]", got, edppMinMu)
	}
}

func TestEDPPCoeffs_Validate(t *testing.T) {
	good := EDPPCoeffs{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: 0}
	if err := good.validate(); err != nil {
		t.Fatalf("good coeffs rejected: %v", err)
	}
	bad := []EDPPCoeffs{
		{AlphaD: 0, AlphaP: 1000, C0: 100, C1: 1, CPf: 10},               // AlphaD must be > 0
		{AlphaD: 1000, AlphaP: 1000, C0: -1, C1: 1, CPf: 10},             // C0 must be >= 0
		{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 10, CAttn: -1}, // CAttn must be >= 0
		{AlphaD: 1000, AlphaP: 2000, C0: 100, C1: 1, CPf: 10},            // AlphaD/AlphaP diverge > 10%
		{AlphaD: 1000, AlphaP: 0, C0: 100, C1: 1, CPf: 10},               // AlphaP must be > 0 (also trips divergence check)
		{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: -1, CPf: 10},           // C1 must be >= 0
		{AlphaD: 1000, AlphaP: 1000, C0: 100, C1: 1, CPf: 0},             // CPf must be > 0
	}
	for i, c := range bad {
		if err := c.validate(); err == nil {
			t.Errorf("case %d: expected validation error, got nil", i)
		}
	}
}
