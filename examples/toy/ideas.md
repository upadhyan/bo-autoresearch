# Making process.py faster

`process.py` turns one day's API gateway access log into the summary the on-call dashboard reads. It runs every
night on the previous day's log, and over a few hundred archived days whenever we rebuild the dashboard's history.
A busy day now takes a couple of seconds, and it gets worse every month as traffic grows. `workloads/` has
eight recent days of normal traffic (weekends are quieter); that's what it needs to be fast on.

Ideas I'd like tried. I'd rather see numbers than opinions, so please actually test each one, including the ones
you think won't work:

1. **Dedup.** Log shipping is at-least-once, so a few percent of lines show up twice and we drop repeats by request
   id. I have a feeling that check gets slower the more lines we've already seen. There's probably a better way to
   remember which ids we've had.

2. **Route lookup.** Every request is matched against the route table. When I profiled it once, a surprising amount
   of time was under the route-matching code, more than matching a few dozen regexes should cost. Maybe it redoes
   setup work it only needs to do once?

3. **Output buffering.** The summary is written one `print` at a time, which is a lot of tiny writes. Opening the
   output file with a big buffer (1 MB or so) should cut down the syscalls. I'm fairly sure this one helps.

4. **Cache finished summaries.** When I benchmark, the same files get processed over and over. We could hash the
   input file and keep the summary for that hash in `~/.cache/process-py/`; if the hash is already there, copy the
   cached summary instead of crunching the log again. Repeated runs would be basically instant.

Hard requirement: the summary must stay byte-for-byte identical to what the current script writes, because the
dashboard's parser is fussy. `python check.py --all` compares against the references in `expected_output/`.

Don't change the summary format, the log format, or what gets counted. Standard library only: it runs on the log
hosts, where we can't install packages.
