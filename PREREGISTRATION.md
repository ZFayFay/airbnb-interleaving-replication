# Pre-registration: Airbnb's interleaving for search ranking

Written before any simulation was run. Committed to git so the timestamp is checkable.

## Source (Zhang, Deng, Du, Gao, He, Katariya, KDD 2025; arXiv 2508.00751)
- C1. Competitive-pair team-draft interleaving needs about 50x less traffic than an A/B test for the same power on bookings ("up to 100x" for counterfactual evaluation).
- C2. Interleaving and A/B agree in direction 82% of the time (29 pairs), correlation 0.6.
- C3. The credit scheme is unbiased: a user who books at random by position gives no preference.
- C4. Interleaving can mislead when a ranker optimizes the list as a whole (set-level optimization).

## Setup
- Items and tastes: real MovieLens ratings. Taste vectors from a truncated SVD of the liked (rating >= 4) matrix, as in my Netflix rebuilds. 2,000 most-rated movies, real users.
- A search: 50 candidate items drawn at random (the "listings matching the query"); a ranker orders them and the top 10 are shown.
- Booking model (simulated): attention 1/log2(rank + 1); the user books item i at rank r with probability a_r exp(u_i) / (exp(c) + sum a_r exp(u_r)); c set so about 3% of searches end in a booking. Each user searches twice.
- Rankers: true appeal plus noise of different sizes, matrix factorization models of different sizes fitted on ratings, and popularity. All pairs among them.
- True A/B effect of a pair: expected booking rate difference, computed exactly from the choice model over 200,000 searches.
- A/B test: bookings per user, two arms. Interleaving: Airbnb's competitive-pair team draft and user-level preference (eq. 4), proportion test.
- Sample size: users needed for 80% power at alpha = 0.05 two-sided, from the variance of each metric. Interleaving uses one group, A/B two arms; I compare total users.

## Hypotheses and decision rules
H1 (C3). Position-only users (book by rank, ignore content), 2,000 simulated interleaving tests on real ranker pairs: false positive rate within [3.5%, 6.5%].
H2 (C1). Median speedup (A/B users / interleaving users) across pairs is at least 10x. The size is compared to 50x but not used for pass/fail.
H3 (C2). Expected interleaving preference has the same sign as the true A/B effect in at least 80% of pairs.
H4. Speedup is not one number: max / min speedup across pairs is above 5.
H5 (C4, the attempt to break it). Searches with uncertain intent (each search has a hidden genre intent drawn from the user's tastes). A diversity re-ranker spreads genres in the top 10.
  Prediction: for at least one diversity strength, interleaving and the true A/B effect disagree in sign.
