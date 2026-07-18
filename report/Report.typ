#import "template/lib.typ" as template

// Document-wide settings
#set quote(block: true)
#set page(numbering: "1")


// Load Template
#show: template.template.with(
  course: [AI Engineering],
  title: [Agentic Dependency Parsing in\ Low-Resource Settings],
  author: (
      name: "Dang Nhat Quang, Han Dai, Nikhila Gadge, Jan Raring",
      department: "Linguistic Data Science Lab",
      institution: "Faculty of Computer Science",
      city: "Bochum",
      country: "Germany",
      mail: "",
  ),
  date: "19 July 2026",
  abstract: [
    In natural language processing (NLP), dependency parsing refers to the task of constructing sentence-level dependency trees. Often, the resulting dependency–head relations are annotated to specify the type of dependency. Large collections of such dependency trees—so called treebanks—are a valuable resource in the study of syntax, grammar, and typology. The manual creation of such dependency trees, however, is a very laborious undertaking which should, if possible, be performed at least semi-automatically. For high-resource languages, many popular NLP libraries (Stanza, NLTK, SpaCy) provide the necessary tooling for exactly this sort of (pre-)annotation. In the case of low-resource languages, however, such tools are typically not available. This observation has instigated us to test whether an AI agent, i.e. an LLM equipped with adequate tools, can function as a general purpose dependency parser, thereby eliminating the need to train purpose-built neural networks as used by the aforementioned libraries.
  ]
)

// Settings for the main matter
#set heading(numbering: "1.1")
#set par(leading: 1.15em, first-line-indent: 1.5em)



// Main Text

#include "chapters/introduction.typ"
#include "chapters/related_work.typ"
#include "chapters/our_approach.typ"
#include "chapters/experiment.typ"
#include "chapters/results.typ"
#include "chapters/conclusion.typ"
#include "chapters/contribution_statement.typ"


// References
#bibliography("refs.bib", style: "apa")

