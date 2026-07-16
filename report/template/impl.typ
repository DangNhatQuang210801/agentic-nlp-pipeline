#let make-venue = move(dy: -2cm, {
  box(pad(left: 10pt, bottom: 10pt,
    image("rub.png", height: 2.5cm)
  ))
})

#let make-title(
  course,
  title,
  author,
  date,
  abstract,
) = {
  set par(spacing: 1.5em)
  set text(font: "TeX Gyre Heros")

  par(
    justify: false,
    text(16pt, fill: rgb("004b71"), course, weight: "bold")
  )
  
  par(
    justify: false,
    leading: 1.05em,
    text(24pt, fill: rgb("004b71"), title, weight: "bold")
  )

  par(
    justify: false,
    (
      text(12pt, [#author.name]),
      text(10pt, [#link("mailto:" + author.mail)]))
      .join("\u{2003}")
  )

  par(
    justify: false,
    text(8pt, "Submission: " + date)
  )

  parbreak()

  v(8pt)
  set text(10pt)
  set par(justify: true, leading: 1.15em)

  [
    #heading(outlined: false, bookmarked: false)[Abstract]
    #text(font: "TeX Gyre Pagella", abstract)
    #v(3pt)
  ]
  v(18pt)
}

#let template(
    course: [],
    title: [],
    author: (),
    date: "",
    doi: "",
    abstract: [],
    make-venue: make-venue,
    make-title: make-title,
    body,
) = {
    set page(
      paper: "a4",
      margin: (top: 1.9cm, bottom: 1in, x: 1.6cm),
      columns: 2
    )
    set par(justify: true)
    set text(10pt, font: "TeX Gyre Pagella")
    set list(indent: 8pt)
    // show link: set text(underline: false)
    show heading: set text(size: 11pt)
    show heading.where(level: 1): set text(font: "TeX Gyre Heros", fill: rgb("004b71"), size: 12pt)
    show heading: set block(below: 8pt)
    show heading.where(level: 1): set block(below: 12pt)

    place(make-venue, top, scope: "parent", float: true)
    place(
      make-title(course, title, author, date, abstract), 
      top, 
      scope: "parent",
      float: true
    )


    show figure: align.with(center)
    show figure: set text(8pt)
    show figure.caption: pad.with(x: 10%)

    // show: columns.with(2)
    body
  }
