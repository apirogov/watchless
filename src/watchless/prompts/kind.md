Append this to the summary prompt when the output is being SCORED rather than
read. It asks the model to classify the video, and the classification is read
mechanically and stripped before any reader sees it.

It is not part of the shipped prompt: an ordinary run never needs the model to
label its own work, and asking costs tokens and invites the model to explain
itself.

Placeholders: none ({n} is not used here).

---

Before writing anything, decide what KIND of video this is. Use the choice to
pick HOW to summarise. Then emit it once, as the FIRST line, in this exact
form, and nothing else on that line:

  KIND: information | performance | both

That line is machine-read and is stripped before the reader sees anything, so
it costs them nothing. Do not mention the classification anywhere else, and do
not explain it -- the reader wants the result, not your working.
- INFORMATION - the words CONVEY INFORMATION that survives leaving the video:
  explainer, how-to, review, report, interview, argument, panel discussion.
  A line of argument or a procedure is still true off-screen, so extract it.
- PERFORMANCE - the words ARE A PERFORMANCE. Music, comedy, sketch, stand-up,
  satire, impersonation, a speech written to be funny, a role-play, a
  dramatic reading. The lines are the artifact: quoting them is not a
  summary, it is a transcript with extra steps, and it destroys the only
  reason to watch. Describe it in 2-4 [overview] bullets instead: what it
  is, who performs it, its subject, its target, its stance, its register --
  and stop.
  A speech or sketch goes in PERFORMANCE even when its jokes are ABOUT politics,
  news or ideas. Being topical does not make it informational; being
  performed does. Do not transcribe punchlines, bits, or quoted jokes.
- BOTH - the words carry information AND there is a visible artifact or
  process to describe (making-of, documentary, demo, build log). Say in one
  [overview] sentence what the video shows, then extract the process facts -- tools,
  versions, asset sources, measurements, timings, techniques. A making-of
  whose bullets contain no tool names or measurements failed. A television
  panel or an interview is not BOTH either: there is nothing to look at.
