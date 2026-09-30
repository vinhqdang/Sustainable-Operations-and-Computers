"""Assemble the submission package in submission/ :

  manuscript/       flat LaTeX source (elsarticle), bibliography, compiled PDF
  figures/          every illustration as a separate file, numbered as in the paper
  Figure_captions.txt   captions supplied separately from the figures (plain text)
  Highlights.docx       3-5 bullets of at most 85 characters, editable
  Declaration_of_interest.docx, Cover_letter.docx
  (Graphical_abstract.* is written by graphical_abstract.py)

Run from src/ after the manuscript compiles: python3 make_submission.py"""
import re, shutil, subprocess, pathlib
from docx import Document
from docx.shared import Pt

ROOT = pathlib.Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"
SUB = ROOT / "submission"
OUT = SUB / "manuscript"
MAIN = "carma_manuscript"

HIGHLIGHTS = [l[2:].strip() for l in open(MS / "highlights.txt") if l.startswith("• ")]
assert 3 <= len(HIGHLIGHTS) <= 5 and all(len(h) <= 85 for h in HIGHLIGHTS), HIGHLIGHTS


def flat_copy():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    tex = [MS / f"{MAIN}.tex"] + sorted(MS.glob("sec_*.tex")) + sorted(MS.glob("appendix_*.tex"))
    need_tables, need_figs = set(), set()
    for f in tex:
        s = open(f).read()
        need_tables |= set(re.findall(r"\\inputtable\{\.\./results/([^}]+)\}", s))
        need_figs |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./figures/([^}]+)\}", s))
        s = re.sub(r"\\inputtable\{\.\./results/([^}]+)\}", r"\\inputtable{\1}", s)
        s = re.sub(r"(\\includegraphics(?:\[[^\]]*\])?)\{\.\./figures/([^}]+)\}", r"\1{\2}", s)
        (OUT / f.name).write_text(s)
    for t in need_tables:
        shutil.copy(ROOT / "results" / t, OUT / t)
    for g in need_figs:
        shutil.copy(ROOT / "figures" / g, OUT / g)
    shutil.copy(MS / "references.bib", OUT / "references.bib")
    return sorted(need_figs)


def build():
    def run(*cmd):
        r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
        return r
    run("pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex")
    b = run("bibtex", MAIN)
    run("pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex")
    run("pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex")
    log = open(OUT / f"{MAIN}.log").read()
    errs = re.findall(r"^! .*", log, flags=re.M)
    undefined = re.findall(r"LaTeX Warning: (?:Reference|Citation) .* undefined", log)
    assert (OUT / f"{MAIN}.pdf").exists(), "no PDF"
    assert not errs and not undefined, (errs[:3], undefined[:3])
    for ext in ("aux", "log", "out", "blg", "spl", "brf"):
        (OUT / f"{MAIN}.{ext}").unlink(missing_ok=True)
    return log


GREEK = {"theta": "θ", "rho": "ρ", "kappa": "κ", "ell": "ℓ", "beta": "β", "sigma": "σ", "eta": "η", "Omega": "Ω", "upsilon": "υ",
         "Pi": "Π", "varepsilon": "ε", "epsilon": "ε", "le": "≤", "ge": "≥", "geq": "≥", "leq": "≤", "times": "×", "approx": "≈",
         "sqrt": "√", "to": "→", "min": "min", "max": "max", "pm": "±"}


def plain(tex, labels=None):
    labels = labels or {}
    t = re.sub(r"\\(?:emph|textbf|textit|texttt)\{([^}]*)\}", r"\1", tex)
    t = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), t)
    t = re.sub(r"\\appref\{([^}]*)\}", lambda m: "Appendix " + labels.get(m.group(1), "?"), t)
    t = re.sub(r"\\cite\{[^}]*\}", "", t)
    t = t.replace("~", " ").replace("\\%", "%").replace("\\,", " ").replace("--", "–").replace("\\_", "_")
    t = re.sub(r"\\(" + "|".join(GREEK) + r")(?![a-zA-Z])", lambda m: GREEK[m.group(1)], t)
    t = re.sub(r"\\(?:hat|bar|tilde)\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\\text\{([^}]*)\}", r"\1", t)
    t = re.sub(r"\$([^$]*)\$", r"\1", t)
    t = re.sub(r"\\[a-zA-Z]+\s*", "", t)
    t = re.sub(r"\s+", " ", t.replace("{", "").replace("}", "")).strip()
    return t.replace("^*", "*")


def figures():
    """Separate figure files named by their number in the paper, plus captions."""
    aux = open(OUT / f"{MAIN}.aux").read() if (OUT / f"{MAIN}.aux").exists() else ""
    return aux


def main():
    SUB.mkdir(exist_ok=True)
    figs = flat_copy()
    # first compile keeps the .aux for figure numbering
    def run(*cmd):
        return subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    run("pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex"); run("bibtex", MAIN)
    run("pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex"); run("pdflatex", "-interaction=nonstopmode", f"{MAIN}.tex")
    aux = open(OUT / f"{MAIN}.aux").read()
    labels = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux))
    figdir = SUB / "figures"
    if figdir.exists():
        shutil.rmtree(figdir)
    figdir.mkdir()
    caps = []
    for f in [MS / f"{MAIN}.tex"] + sorted(MS.glob("sec_*.tex")) + sorted(MS.glob("appendix_*.tex")):
        s = open(f).read()
        for blk in re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", s, flags=re.S):
            g = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{\.\./figures/([^}]+)\}", blk).group(1)
            lab = re.search(r"\\label\{([^}]+)\}", blk).group(1)
            cap = re.search(r"\\caption\{(.*)\}\s*\\label", blk, flags=re.S).group(1)
            n = labels[lab].replace(".", "_")
            shutil.copy(ROOT / "figures" / g, figdir / f"Fig_{n}_{pathlib.Path(g).stem.split('_', 1)[1]}.pdf")
            caps.append((labels[lab], plain(cap, labels)))
    caps.sort(key=lambda c: (c[0][0].isalpha(), c[0].zfill(6) if c[0].isdigit() else c[0]))
    (SUB / "Figure_captions.txt").write_text("\n\n".join(f"Fig. {n}. {c}" for n, c in caps) + "\n")
    log = build()
    shutil.copy(OUT / f"{MAIN}.pdf", SUB / "Manuscript.pdf")

    # Word documents
    d = Document(); d.add_heading("Highlights", level=1)
    for h in HIGHLIGHTS:
        d.add_paragraph(h, style="List Bullet")
    d.save(SUB / "Highlights.docx")

    d = Document(); d.add_heading("Declaration of interests", level=1)
    d.add_paragraph("The author declares that he has no known competing financial interests or personal relationships that "
                    "could have appeared to influence the work reported in this paper.")
    d.add_paragraph("Quang-Vinh Dang, British University Vietnam (ORCID 0000-0002-3877-8024)")
    d.save(SUB / "Declaration_of_interest.docx")

    title = re.search(r"\\title\{(.*?)\\tnoteref", open(MS / f"{MAIN}.tex").read(), flags=re.S).group(1)
    d = Document()
    for line in ["Dear Editor-in-Chief,", "",
                 f"Please consider the enclosed manuscript, \"{title}\", for publication as a full-length article in "
                 "Sustainable Operations and Computers.", "",
                 "The paper studies how to schedule deferrable computing jobs across data-center sites so as to reduce "
                 "carbon emissions when low-carbon capacity is scarce and future jobs are unknown. It formulates the problem "
                 "as a minimum-cost flow, proves a lower bound on the competitive ratio that follows from demand uncertainty "
                 "alone, and studies an anticipatory receding-horizon policy (CARMA) whose guarantees hold under stated "
                 "assumptions. The policy is evaluated out of sample on five grids (Great Britain, Europe, the United States, "
                 "Brazil and a four-continent fleet), on a workload derived from a production cluster trace, and against "
                 "strong baselines including a scenario-based stochastic lookahead. The paper also reports unfavorable "
                 "results: limited benefit without migration, a small advantage under a marginal-emission proxy, and a "
                 "stochastic lookahead that is slightly better at a much higher computing cost. Carbon attribution among "
                 "sites is treated as a cooperative game, with a dual-price allocation that lies in the core.", "",
                 "The topic fits the scope of the journal on low-carbon operations, optimization and computer-based decision "
                 "support, and it builds on work published in it.", "",
                 "This work has not been published previously, is not under consideration elsewhere, and is approved by the "
                 "sole author. The data come from public sources, and all code and raw results are available from the "
                 "repository named in the Data availability statement. I declare no competing interests.", "",
                 "Sincerely,", "Quang-Vinh Dang", "British University Vietnam, Hung Yen, Vietnam",
                 "vinh.dq4@buv.edu.vn  |  ORCID 0000-0002-3877-8024"]:
        d.add_paragraph(line)
    d.save(SUB / "Cover_letter.docx")
    print("package ready:", sorted(p.name for p in SUB.iterdir()))
    print(len(list(figdir.iterdir())), "figure files;", "manuscript OK")


if __name__ == "__main__":
    main()
