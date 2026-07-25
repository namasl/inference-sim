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

// Wp is the prefill demand of a_p uncached tokens for a prompt of full length a_r, in
// µs. It is the trajectory sum of the active (trained-physics) latency model's causal
// per-step prefill charge C_pf·s + C_attn·s·(prefix + s/2), integrated over the prefill
// from prefix a_r−a_p to a_r — hence the (a_r − a_p/2) form. At a_p = a_r (no cache) this
// is C_pf·a_r + 0.5·C_attn·a_r².
func (c EDPPCoeffs) Wp(ap, ar int) float64 {
	a := float64(ap)
	r := float64(ar)
	return c.CPf*a + c.CAttn*a*(r-a/2.0)
}

// Wd is the decode demand for a prompt of length a_r generating o output tokens,
// in µs: the exact discrete per-step sum Σ_{k=0}^{o-1}(C0 + C1·(a_r+k)) =
// C0·o + C1·o·(a_r + (o-1)/2). Matches the active latency model's per-decode-step
// charge (context = ProgressIndex = a_r + k). o is the N̂_out estimate at routing.
func (c EDPPCoeffs) Wd(ar int, o float64) float64 {
	if o <= 0 {
		return 0
	}
	r := float64(ar)
	return c.C0*o + c.C1*o*(r+(o-1)/2.0)
}

// tIterDecode is the E3 decode iteration time (µs) at the given batch state.
func (c EDPPCoeffs) tIterDecode(bDec int, kv, sPf int64) float64 {
	return c.AlphaD + c.C0*float64(bDec) + c.C1*float64(kv) + c.CPf*float64(sPf)
}

// tIterPrefill is the E3 prefill iteration time (µs); a dedicated prefill server
// runs no decode work (B_dec = 0, KV = 0).
func (c EDPPCoeffs) tIterPrefill(sPf int64) float64 {
	return c.AlphaP + c.CPf*float64(sPf)
}

// muDecode / muPrefill are the live drain rates μ = 1 − α/T_iter (design §4),
// clamped to [edppMinMu, 1] so predictor denominators never collapse.
func (c EDPPCoeffs) muDecode(bDec int, kv, sPf int64) float64 {
	return clampMu(1.0 - c.AlphaD/c.tIterDecode(bDec, kv, sPf))
}

func (c EDPPCoeffs) muPrefill(sPf int64) float64 {
	return clampMu(1.0 - c.AlphaP/c.tIterPrefill(sPf))
}

// muDNom is the fixed nominal decode drain rate at the SLO-critical batch
// (T_iter = τ_itl), design §7. Caller guarantees τ_itl > AlphaD (config guard).
func (c EDPPCoeffs) muDNom(tauITLUs float64) float64 {
	return clampMu(1.0 - c.AlphaD/tauITLUs)
}

// muPNom is the fixed nominal prefill drain rate at the nominal operating chunk
// S_pf^nom (design §7).
func (c EDPPCoeffs) muPNom(sPfNom int) float64 {
	return clampMu(1.0 - c.AlphaP/(c.AlphaP+c.CPf*float64(sPfNom)))
}

// deltaBarDecode is the marginal decode work per step at context length ctxLen
// (design §6.3: δ̄_dec = c0 + c1·L̄), in µs.
func (c EDPPCoeffs) deltaBarDecode(ctxLen float64) float64 {
	return c.C0 + c.C1*ctxLen
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
