package sim

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// EDPPCoeffs are the frozen E3 latency-law coefficients (design §1.1). Units:
// AlphaD/AlphaP in µs, C0 in µs/req, C1/CPf in µs/token, CAttn in µs/token².
type EDPPCoeffs struct {
	AlphaD float64 // α: decode per-iteration fixed cost
	AlphaP float64 // α_p: prefill per-iteration fixed cost (≈ AlphaD)
	C0     float64 // decode per-request overhead
	C1     float64 // decode KV-read per resident token
	CPf    float64 // exposed prefill compute per token (= k_p)
	CAttn  float64 // prefill attention term
}

// edppCoeffsJSON mirrors the nested shape of scripts/calibration/coeffs-*.json.
type edppCoeffsJSON struct {
	Decode struct {
		AlphaUs      float64 `json:"alpha_us"`
		C0UsPerReq   float64 `json:"c0_us_per_req"`
		C1UsPerToken float64 `json:"c1_us_per_token"`
	} `json:"decode"`
	Prefill struct {
		AlphaPUs       float64 `json:"alpha_p_us"`
		CPfUsPerToken  float64 `json:"c_pf_us_per_token"`
		CAttnUsPerUnit float64 `json:"c_attn_us_per_unit"`
	} `json:"prefill"`
}

// LoadEDPPCoeffs reads a frozen-coefficients JSON file and returns validated
// coefficients. A missing/unreadable file or invalid coefficients is an error
// (the CLI boundary turns it into logrus.Fatalf; R3).
func LoadEDPPCoeffs(path string) (EDPPCoeffs, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return EDPPCoeffs{}, fmt.Errorf("edpp coeffs: read %q: %w", path, err)
	}
	var j edppCoeffsJSON
	if err := json.Unmarshal(raw, &j); err != nil {
		return EDPPCoeffs{}, fmt.Errorf("edpp coeffs: parse %q: %w", path, err)
	}
	c := EDPPCoeffs{
		AlphaD: j.Decode.AlphaUs,
		AlphaP: j.Prefill.AlphaPUs,
		C0:     j.Decode.C0UsPerReq,
		C1:     j.Decode.C1UsPerToken,
		CPf:    j.Prefill.CPfUsPerToken,
		CAttn:  j.Prefill.CAttnUsPerUnit,
	}
	if err := c.validate(); err != nil {
		return EDPPCoeffs{}, fmt.Errorf("edpp coeffs %q: %w", path, err)
	}
	return c, nil
}

// validate enforces positivity of the fixed costs and non-negativity of the
// per-token coefficients, plus an α ≈ α_p sanity bound (design §1.1: prefill and
// decode share the same per-iteration intercept; > 10% divergence means the JSON
// was fit on mismatched hardware/regimes).
func (c EDPPCoeffs) validate() error {
	switch {
	case c.AlphaD <= 0:
		return fmt.Errorf("AlphaD must be > 0, got %v", c.AlphaD)
	case c.AlphaP <= 0:
		return fmt.Errorf("AlphaP must be > 0, got %v", c.AlphaP)
	case c.C0 < 0:
		return fmt.Errorf("C0 must be >= 0, got %v", c.C0)
	case c.C1 < 0:
		return fmt.Errorf("C1 must be >= 0, got %v", c.C1)
	case c.CPf <= 0:
		return fmt.Errorf("CPf must be > 0, got %v", c.CPf)
	case c.CAttn < 0:
		return fmt.Errorf("CAttn must be >= 0, got %v", c.CAttn)
	}
	if rel := math.Abs(c.AlphaD-c.AlphaP) / c.AlphaD; rel > 0.10 {
		return fmt.Errorf("AlphaD (%v) and AlphaP (%v) diverge by %.1f%% (> 10%%)", c.AlphaD, c.AlphaP, rel*100)
	}
	return nil
}
