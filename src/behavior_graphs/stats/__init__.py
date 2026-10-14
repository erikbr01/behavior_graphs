"""Statistical comparison of policies from rollout outcomes.

Kept out of ``model_selection`` on purpose: that package ranks *pipeline
configurations* on clustering metrics, whereas this one compares *policies*
on task success, which is a binomial-proportion problem with its own tests.
"""
