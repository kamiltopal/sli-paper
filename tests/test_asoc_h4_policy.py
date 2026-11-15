"""Focused Phase 5 leakage, grouping, fairness, and utility tests."""
import unittest

import numpy as np
import pandas as pd

from scripts.asoc_h4_policy import (archived_decisions, matched_decisions,
    reference_from_calibration, predecision_features, h4_select,
    evaluate_seed_blocks, report_tables)


def fake_archived():
    b, a = [], []
    for seed in range(3):
        file = f"om0.5_plain_s{seed}.npy"
        for regime in (0.1, 1.0):
            for rep in (0, 1):
                b.append(dict(file=file, regime=regime, arch="dlinear", rep=rep,
                              val_mse=0.2 + 0.01 * seed + 0.1 * regime,
                              test_mse=0.2 + 0.01 * seed + 0.1 * regime))
                for gen, pct in (("vae", -5+seed), ("bootstrap", 7+seed),
                                 ("exog", -3), ("seasonal", -10)):
                    a.append(dict(file=file, regime=regime, arch="dlinear", rep=rep,
                                  generator=gen, test_mse=b[-1]["test_mse"] * (1+pct/100)))
    return pd.DataFrame(b), pd.DataFrame(a)


class PolicyTests(unittest.TestCase):
    def test_archived_excludes_oracle_by_default(self):
        b, a = fake_archived()
        df = archived_decisions(b, a)
        self.assertEqual(len(df), 3 * 2 * 3)
        self.assertNotIn("seasonal", set(df.generator))
        self.assertEqual(set(df.openness), {0, 1})
        self.assertEqual(len(archived_decisions(b, a, include_oracle=True)), 24)

    def test_validation_reference_from_calibration_only(self):
        b, a = fake_archived()
        df = archived_decisions(b, a)
        cal = df[df.series_seed != 2]
        held = df[df.series_seed == 2]
        r = reference_from_calibration(cal)
        f = predecision_features(held, r)
        other = held.copy()
        other["base_test_mse"] = 99999
        other["aug_test_mse"] = .00001
        other["delta_pct"] = -99
        other["target_omega"] = -1
        f2 = predecision_features(other, r)
        np.testing.assert_array_equal(f.headroom.to_numpy(), f2.headroom.to_numpy())
        np.testing.assert_array_equal(h4_select(f, 0.2), h4_select(f2, 0.2))

    def test_fold_policy_does_not_peek_at_heldout_test(self):
        b, a = fake_archived()
        df = archived_decisions(b, a)
        d0, f0 = evaluate_seed_blocks(df)
        altered = df.copy()
        altered.loc[altered.series_seed == 0, "delta_pct"] = 1234.0
        d1, f1 = evaluate_seed_blocks(altered)
        original = d0[(d0.series_seed == 0) & (d0.policy == "h4_validation")]
        changed = d1[(d1.series_seed == 0) & (d1.policy == "h4_validation")]
        np.testing.assert_array_equal(original.selected_augment, changed.selected_augment)
        self.assertEqual(float(f0.loc[f0.heldout_series_seed == 0,
                                      "threshold_validation_headroom"].iloc[0]),
                         float(f1.loc[f1.heldout_series_seed == 0,
                                      "threshold_validation_headroom"].iloc[0]))

    def test_folds_hold_out_all_same_seed_series(self):
        b, a = fake_archived()
        b2 = b.copy(); a2 = a.copy()
        b2["file"] = b2.file.str.replace("om0.5", "om0.8")
        a2["file"] = a2.file.str.replace("om0.5", "om0.8")
        df = archived_decisions(pd.concat([b,b2]), pd.concat([a,a2]))
        d, folds = evaluate_seed_blocks(df)
        self.assertEqual(folds.heldout_decisions.tolist(), [12,12,12])
        self.assertEqual(len(d), 36 * 5)
        # Within a fold the same seed's different Omega series are held out together.
        self.assertEqual(d[d.series_seed==0].file.nunique(),2)

    def test_matched_requires_equal_steps_and_completed_pair(self):
        rows=[]
        for seed in range(3):
            for condition, mse in [("duplicate_real", .10), ("vae_augmented", .09)]:
                rows.append(dict(file=f"om0.5_plain_s{seed}.npy", regime=.1,
                                 arch="dlinear", input_mode="target", rep=0, pool_seed=seed,
                                 epochs=10, vae_epochs=30, batch_size=64, lr=.001,
                                 condition=condition, test_mse=mse, val_mse=mse+.001,
                                 optimizer_steps=410))
        raw = pd.DataFrame(rows)
        v = matched_decisions(raw)
        self.assertEqual(len(v),3)
        self.assertEqual(v.generator.unique().tolist(),["vae"])
        damaged = raw.copy()
        damaged.loc[damaged.condition == "vae_augmented", "optimizer_steps"] += 1
        with self.assertRaisesRegex(ValueError,"Unequal optimizer"):
            matched_decisions(damaged)
        with self.assertRaisesRegex(ValueError,"Unmatched"):
            matched_decisions(raw.iloc[:-1])

    def test_utility_oracle_regret_and_sign(self):
        b, a = fake_archived()
        df=archived_decisions(b,a)
        decisions,_=evaluate_seed_blocks(df)
        self.assertTrue((decisions.regret_pct_points >= -1e-10).all())
        never=decisions[decisions.policy=="never"]
        self.assertTrue((never.realized_delta_pct==0).all())
        oracle=decisions[decisions.policy=="oracle_test_only"]
        self.assertTrue((oracle.regret_pct_points==0).all())
        self.assertTrue((oracle.correct_sign_choice).all())
        per,summ=report_tables(decisions)
        self.assertEqual(len(summ),5)
        self.assertEqual(set(summ.n_heldout_seed_blocks),{3})
        self.assertEqual(len(per),15)

    def test_no_reference_from_heldout_validation(self):
        b,a=fake_archived()
        df=archived_decisions(b,a)
        refs=reference_from_calibration(df[df.series_seed<2])
        self.assertAlmostEqual(refs[("dlinear",.1)],.21)
        # Even artificially changing held-out validation does not alter the ref.
        df.loc[df.series_seed==2,"base_val_mse"] = .00001
        self.assertEqual(refs, reference_from_calibration(df[df.series_seed<2]))


if __name__ == "__main__":
    unittest.main()
