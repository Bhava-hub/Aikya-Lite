# NOTE — how PCA works, what V14 means, and why this dataset uses it
#
# HOW PCA FINDS "VARIATION":
# Imagine every transaction plotted as a point using the original real
# features (amount, hour, location, etc.) as axes. The cloud of points
# isn't necessarily spread evenly along those original axes — it might
# actually stretch out diagonally, mixing several features at once.
# PCA's job is to find the exact directions where the data spreads out
# the most. It does this by building a covariance matrix (a table showing
# how much every pair of features moves together), then mathematically
# solving for the best directions through that data (called eigenvectors),
# each with a score (eigenvalue) showing how much spread it captures.
# The direction with the most spread becomes V1, the next-most spread
# (while staying mathematically independent of V1) becomes V2, and so on
# down through V28, which captures very little spread at all.
#
# WHAT V14 SPECIFICALLY MEANS:
# Each component (V1, V2, ... V28) is really just a "recipe" — a list of
# weights, one per original real feature, describing how to blend all of
# them together into one new number. V14 is one specific recipe/blend;
# V3 is a completely different blend of the SAME original features. A
# transaction's "V14 value" is just: take that transaction's real feature
# values, apply V14's specific weights, and sum them up. A "high" V14
# value means this transaction's blend-result is far from the typical
# blend-result most transactions get — i.e., something about its
# underlying (real, hidden) features is unusual compared to most
# transactions. That's why an unusual V14 value can correlate with fraud:
# fraud tends to look different from normal behavior along some
# combination of the real, original attributes.
#
# WHY PCA IS USED HERE, SPECIFICALLY:
# The bank that owns this data could not legally publish real columns
# (merchant, location, account details, etc.). PCA is a one-way
# mathematical scramble — the exact weight recipes were never published,
# so V1-V28 cannot be reversed back into the real features. Releasing
# PCA components instead of raw data protects customer privacy while
# still preserving genuine fraud patterns for research/modeling.
#
# WHY PCA IS USED IN ML GENERALLY (separate reason, not really why it's
# used here):
# Beyond privacy, PCA is commonly used to shrink a large number of noisy
# or redundant raw features down to a smaller set of components that
# still capture most of the meaningful variation — this speeds up
# training, reduces the data needed, and can remove low-signal noise
# (the last few components, like V27-V28, capture almost nothing and are
# often just noise). It also removes correlation between features, since
# PCA components are mathematically guaranteed to be independent of each
# other. This dataset only has 30 features to begin with though, so this
# compression benefit isn't really the motivation here — privacy is.
#
# CONSEQUENCE FOR THIS PROJECT:
# Because we only have the anonymized components, SHAP can only say
# things like "V14 drove this fraud prediction," not something
# interpretable like "unusual transaction location drove this
# prediction." The explainability mechanism itself still works correctly
# — it's just explaining scrambled columns instead of real ones. In a
# real deployment on a bank's actual, non-anonymized data, PCA would be
# skipped entirely, and SHAP would point directly to real, actionable
# features instead of V-numbers.