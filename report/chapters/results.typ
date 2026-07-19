#import "../template/lib.typ" as template

= Results

After running the dependency parsing experiment, the aforementioned metrics have been computed and compiled into a CSV-file (`data/processed/parse.csv`) for further analysis.

#figure(
  image("../plots/parse_success.png"),
  caption: [Difference in UAS by language and tool use.]
) <parse_success>

In the experiment the model was restricted to being able to generate a maximum of 10,000 new tokens per iteration. Maxing out this quota was a common reason for not producing a final answer. @parse_success shows the rate of success for parsing a dependency tree from the models output. One can see that the run with tools drastically increased the successrate—expecially for English and Low Saxon. This is not totally surprising since the average sentence length for these languages were generally higher, therefore demanding more reasoning. Also, after every tool call the token limit was reset, to the effect that with tools, the model was able to go way beyond 10,000 tokens. However, the number of generated tokens never surpassed 11,678, indicating that, perhaps, tool calls did help the model commit to its intermediate results more easily. Indeed, the average number of generated tokens dropped sharply from 7801 (without tools) to just 4980 (with tools).

#figure(
  image("../plots/tree_validity.png"),
  caption: [Ratio of dependency graphs having a valid tree structure.]
) <tree_validity>

As shown in @tree_validity, a symptom of the reduced reasoning effort is a reduction in the ability to produce formally valid trees, i.e. trees that are connected and acyclic, and whose `HEAD`-ids are all in the range of token ids. While the model did not make a single mistake in the setting without tools, _with_ tools it became sloppy when it came to longer sentences; and that even though it had a tree validation tool at its disposal.

#figure(
  image("../plots/uas_language_tools.png"),
  caption: [Overall UAS by language and tool use.]
) <uas_general>

Interestingly, things are the other way round when it comes to the most important metric, the unlabeled attachment score (UAS), which measures the ratio of correctly predicted edges. For Taiwanese the average UAS dropped from 78% (overall the highest score measured) to 68%. For Marathi and Vietnamese, the score stayed more or less stable at around 55% for both languages. Only the English and Low Saxon scores improved by a noticeable margin from 19% to 35% and 23% to 31% respectively (compare @uas_table and @uas_general).

#set table(
  stroke: none,
  gutter: 0.2em,
  fill: (x, y) =>
    if x == 0 or y == 0 { gray },
  inset: (right: 1.5em),
)
#show table: set text(size: 8pt)
#show table.cell: it => {
  if it.x == 0 or it.y == 0 {
    set text(white)
    strong(it)
  } else if it.body == [] {
    // Replace empty cells with 'N/A'
    pad(..it.inset)[_N/A_]
  } else {
    it
  }
}
#let green_cell(body) = table.cell(
  fill: green.lighten(60%),
)[#body]
#let red_cell(body) = table.cell(
  fill: red.lighten(60%),
)[#body]
#let yellow_cell(body) = table.cell(
  fill: yellow.lighten(60%),
)[#body]

#figure(
  table(
    columns: 4,
    [Language], [UAS Without Tools], [UAS With Tools], [Change],
    [English], [0.1882], [0.3355], green_cell[+78.3%],
    [Marathi], [0.5542], [0.5569], yellow_cell[+0.5%],
    [Taiwanese], [0.7833], [0.6817], red_cell[-13.0%],
    [Low Saxon], [0.2282], [0.3127], green_cell[+37.0%],
    [Vietnamese], [0.5235], [0.5499], yellow_cell[+5.0%],
  ),
  caption: [Per-language change in UAS by modality.]
) <uas_table>

Finally, as can be seen in @uas_diff, the per-sentence difference in UAS is in good alignment with the differences in the overall per-language UAS scores.

#figure(
  image("../plots/uas_diff_language.png"),
  caption: [Per-sentence difference in UAS by language and tool use.]
) <uas_diff>


