# Is a run reproducible?

## In short

*Waiting for the runs.*

## Thesis

```thesis E01
```

Everything after this chapter compares runs with each other. That only means
something if a run is decided by its settings and its seed and by nothing
else — not by when it was stopped, not by a crash, not by how many threads the
computer happened to give it.

## Method

```experiment E01
```

Every run starts from the baseline B1 with 10,000 tokens, which grows to a
few thousand agents within the 30 iterations. Each of three seeds is run four
ways:

- **straight** — from start to iteration 30 in one go. This is the reference.
- **stopped at 10** — stopped at iteration 10, saved, and continued later in a
  *new* process, from the checkpoint on disk.
- **cut off at 15** — the process dies at iteration 15 without saving, the way
  it would if the power went. Its last checkpoint is from iteration 10, so it
  resumes from there and lives iterations 10 to 15 a second time.
- **all BLAS threads** — the matrix library may use every core. Normally the
  lab keeps it to one thread per run.

For each variant, every recorded frame, every row of statistics and the final
checkpoint — every array of it, including the state of the random number
generator — is compared with the straight run of the same seed.

## Results

*Waiting for the runs.*

## Conclusion

*Waiting for the runs.*
