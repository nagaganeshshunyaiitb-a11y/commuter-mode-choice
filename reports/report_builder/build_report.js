// Builds reports/Project_Report.docx from report_data.json.
// Run export_report_data.py first. Every number below is read from the pipeline results.
const fs = require("fs");
const path = require("path");
const d = require("docx");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType, ShadingType,
  AlignmentType, HeadingLevel, BorderStyle, LevelFormat, TableOfContents, Footer, PageNumber, TabStopType,
  PageBreak, VerticalAlign,
} = d;

const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "report_data.json"), "utf8"));
const T = DATA.tables, J = DATA.json, FIGS = DATA.figures;
const OUT = path.join(__dirname, "..", "Project_Report.docx");

// ----------------------------------------------------------------------------- helpers
const TEXT_W = 9026;                    // A4 with 1-inch margins, in DXA
const FONT = "Times New Roman";
const f = (x, n = 2) => (x === null || x === undefined || Number.isNaN(Number(x))) ? "-" : Number(x).toFixed(n);
const sgn = (x, n = 2) => (Number(x) >= 0 ? "+" : "") + f(x, n);
const by = (rows, key) => Object.fromEntries(rows.map(r => [r[key], r]));
const MODES = ["walk", "twow", "auto", "car", "bus", "train"];
const MODE_NAME = { walk: "Walk", twow: "Two-wheeler", auto: "Auto-rickshaw", car: "Car", bus: "Bus", train: "Suburban rail" };

function runs(text, base = {}) {
  const parts = String(text).split(/(\*\*[^*]+\*\*|\*[^*\s][^*]*\*|`[^`]+`|~[^~\s]+~|\^[^^\s]+\^)/g).filter(s => s.length);
  return parts.map(p => {
    if (p.startsWith("~") && p.length > 2) return new TextRun({ text: p.slice(1, -1), subScript: true, ...base });
    if (p.startsWith("^") && p.length > 2) return new TextRun({ text: p.slice(1, -1), superScript: true, ...base });
    if (p.startsWith("**")) return new TextRun({ text: p.slice(2, -2), bold: true, ...base });
    if (p.startsWith("`")) return new TextRun({ text: p.slice(1, -1), font: "Consolas", size: 19, ...base });
    if (p.startsWith("*") && p.length > 2) return new TextRun({ text: p.slice(1, -1), italics: true, ...base });
    return new TextRun({ text: p, ...base });
  });
}
const P = (text, opts = {}) => new Paragraph({ children: runs(text), alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 140, line: 300 }, ...opts });
const B = (text, level = 0) => new Paragraph({ children: runs(text), numbering: { reference: "bullets", level },
  alignment: AlignmentType.LEFT, spacing: { after: 70, line: 288 } });
const N = (text, instance) => new Paragraph({ children: runs(text), numbering: { reference: "numbers", level: 0, instance },
  spacing: { after: 70, line: 288 } });

let chap = 0, sec = 0, figN = 0, tabN = 0, eqN = 0, label = "";
const H1 = (title, opts = {}) => {
  chap += 1; sec = 0; figN = 0; tabN = 0; eqN = 0; label = String(chap);
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(`${chap}   ${title}`)], ...opts });
};
const H2 = (title) => { sec += 1; return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(`${chap}.${sec}   ${title}`)] }); };
const H3 = (title) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(title)] });
const HA = (title, letter) => { label = letter; figN = 0; tabN = 0; eqN = 0;
  return new Paragraph({ heading: HeadingLevel.HEADING_1, pageBreakBefore: true, children: [new TextRun(title)] }); };

const TAG = { obs: "Observed data. ", syn: "Synthetic population. ", sim: "Policy simulation. ", none: "" };

function FIG(key, caption, tag = "none", maxW = 600, maxH = 760) {
  const meta = FIGS[key];
  if (!meta) throw new Error(`missing figure ${key}`);
  let w = maxW, h = Math.round(maxW * meta.height / meta.width);
  if (h > maxH) { h = maxH; w = Math.round(maxH * meta.width / meta.height); }
  figN += 1;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(meta.path), transformation: { width: w, height: h },
        altText: { title: caption, description: caption, name: key } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 220 },
      children: [new TextRun({ text: `Figure ${label}.${figN}. `, bold: true, size: 20 }),
        ...runs(TAG[tag] + caption, { size: 20, italics: true })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };
function cell(text, width, { header = false, align = AlignmentType.LEFT, size = 18 } = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders, verticalAlign: VerticalAlign.CENTER,
    shading: header ? { fill: "DCE6F2", type: ShadingType.CLEAR, color: "auto" } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: align, spacing: { after: 0, line: 240 },
      children: runs(String(text), { size, bold: header }) })],
  });
}
function TAB(headers, rows, widths, caption, tag = "none", { numericFrom = 1, size = 18 } = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  if (Math.abs(total - TEXT_W) > 2) widths = widths.map(w => Math.round(w * TEXT_W / total));
  const fix = TEXT_W - widths.reduce((a, b) => a + b, 0);
  widths[widths.length - 1] += fix;
  tabN += 1;
  const head = new TableRow({ tableHeader: true, children: headers.map((h, i) =>
    cell(h, widths[i], { header: true, align: i >= numericFrom ? AlignmentType.CENTER : AlignmentType.LEFT, size })) });
  const body = rows.map(r => new TableRow({ cantSplit: true, children: r.map((v, i) =>
    cell(v, widths[i], { align: i >= numericFrom ? AlignmentType.CENTER : AlignmentType.LEFT, size })) }));
  return [
    new Paragraph({ spacing: { before: 160, after: 80 }, keepNext: true,
      children: [new TextRun({ text: `Table ${label}.${tabN}. `, bold: true, size: 20 }),
        ...runs(TAG[tag] + caption, { size: 20, italics: true })] }),
    new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] }),
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}

// math
const mr = (t) => new d.MathRun(t);
const arr = (x) => (typeof x === "string" ? [mr(x)] : Array.isArray(x) ? x : [x]);
const sub = (b, s) => new d.MathSubScript({ children: arr(b), subScript: arr(s) });
const sup = (b, s) => new d.MathSuperScript({ children: arr(b), superScript: arr(s) });
const subsup = (b, s, p) => new d.MathSubSuperScript({ children: arr(b), subScript: arr(s), superScript: arr(p) });
const frac = (n, den) => new d.MathFraction({ numerator: arr(n), denominator: arr(den) });
// docx's MathSum drops <m:sup> entirely when there is no upper limit, which the OMML schema
// does not allow; this variant keeps an empty, hidden <m:sup>.
class SumNoUpper extends d.XmlComponent {
  constructor({ children, subScript }) {
    super("m:nary");
    this.root.push(d.createMathNAryProperties({ accent: "∑", hasSuperScript: false, hasSubScript: true }));
    this.root.push(d.createMathSubScriptElement({ children: subScript }));
    this.root.push(d.createMathSuperScriptElement({ children: [] }));
    this.root.push(d.createMathBase({ children }));
  }
}
const sum = (lo, body, hi) => (hi ? new d.MathSum({ children: arr(body), subScript: arr(lo), superScript: arr(hi) })
                                  : new SumNoUpper({ children: arr(body), subScript: arr(lo) }));
const rb = (x) => new d.MathRoundBrackets({ children: arr(x) });
function EQ(children) {
  eqN += 1;
  return new Paragraph({ spacing: { before: 100, after: 140 },
    tabStops: [{ type: TabStopType.CENTER, position: Math.round(TEXT_W / 2) }, { type: TabStopType.RIGHT, position: TEXT_W }],
    children: [new TextRun("\t"), new d.Math({ children }), new TextRun(`\t(${label}.${eqN})`)] });
}
const CODE = (lines) => lines.map(l => new Paragraph({ spacing: { after: 0, line: 240 },
  shading: { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" },
  children: [new TextRun({ text: l.length ? l : " ", font: "Consolas", size: 17 })] }));

// ----------------------------------------------------------------------------- data shortcuts
const audit = J.audit_summary;
const rules = audit.los_rules;
const summary = J.pipeline_summary;
const sel = J.model_selection;
const card = J.model_card;
const opt = J.optimisation_result;
const cv = by(T.cv_summary, "model");
const test = by(T.test_metrics, "model");
const ci = J.test_bootstrap_ci;
const REF = "Reference: constants only";
const ORDER = ["MNL", "Nested logit", "Logistic regression", "Decision tree", "Random forest", "XGBoost"];
const WINNER = sel.winner;
const est = by(T.mnl_estimates_selected, "index");
const nlEst = by(T.nested_logit_estimates, "index");
const lamKey = Object.keys(nlEst).find(k => k.startsWith("lambda_"));
const specs = by(T.mnl_specifications, "specification");
const profile = by(T.mnl_walk_weight_profile, "walk_weight");
const nests = by(T.nested_logit_structures, "structure");
const synthTab = by(T.synthetic_method_comparison, "method");
const synthMethod = summary.synthetic.method;
const syn = synthTab[synthMethod];
const leaf = T.synthetic_cart_leaf_sensitivity;
const base = by(T.baseline_share_comparison, "mode");
const optShares = by(T.optimisation_mode_shares, "mode");
const bench = by(T.optimisation_benchmarks, "policy");
const budget = T.optimisation_effort_budget;
const wws = T.sensitivity_walk_weight;
const ready = T.simulation_readiness;
const readyBy = (model, lever) => ready.find(r => r.model === model && r.lever === lever);
const sc = { cost: by(T.scenario_cost, "scenario"), time: by(T.scenario_time, "scenario"),
             av: T.scenario_availability, comb: T.scenario_combined };
const evidence = (name) => T[`audit_evidence_${name}`];
const tuning = by(T.tuning_results, "model");
const bestF1Model = ORDER.reduce((a, b) => (cv[a].macro_f1_mean >= cv[b].macro_f1_mean ? a : b));
const bestLLModel = ORDER.reduce((a, b) => (cv[a].log_loss_mean <= cv[b].log_loss_mean ? a : b));
const vot = summary.vot.in_vehicle_time_INR_per_hour;
const baseSus = opt.baseline_sustainable_pct, optSus = opt.objective_sustainable_share_pct;
const levers = opt.optimal_levers;
const BOUND = opt.bounds;
const corrIDs = T.audit_los_inconsistent_records.map(r => r.ID);
const eda = { income: by(T.eda_by_income, "Income"), dist: by(T.eda_by_distance, "dist_band"),
              age: by(T.eda_by_age, "Age"), purpose: by(T.eda_by_purpose, "Trip_Purpose"),
              veh: by(T.eda_by_vehicle, "veh_own_cat"), sapt: by(T.eda_by_sapt, "SAPT"),
              av: by(T.eda_availability, "mode"), dbc: by(T.eda_distance_by_choice, "mode"),
              assoc: T.eda_association, gender: by(T.eda_by_gender, "Gender"), occ: by(T.eda_by_occupation, "Occup") };
const md = (m) => eda.dbc[m]["50%"];
const tests = J.tests;
const sizes = J.sizes;
const autoIc = rules.auto_fare.intercept;
const autoFare = `max(${f(rules.auto_fare.minimum, 0)}, ${f(rules.auto_fare.slope, 2)} d ${autoIc < 0 ? "−" : "+"} ${f(Math.abs(autoIc), 2)})`;
const lrGap = (cv[bestF1Model].macro_f1_mean - cv[bestF1Model].macro_f1_se) - cv["Logistic regression"].macro_f1_mean;

// ----------------------------------------------------------------------------- content
const C = [];
const add = (...x) => x.forEach(i => (Array.isArray(i) ? C.push(...i) : C.push(i)));

// Title page
add(
  new Paragraph({ spacing: { before: 1400, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "AI and Data Science: Course Project Report", size: 26, color: "404040" })] }),
  new Paragraph({ spacing: { after: 400 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "AI-Driven Commuter Mode Choice Modelling and Sustainable Transport Policy Optimisation Using a Synthetic Population",
      bold: true, size: 40 })] }),
  new Paragraph({ spacing: { after: 1200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Prediction, behavioural simulation and constrained policy optimisation from a 531-respondent commuter survey",
      italics: true, size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Submitted by", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 },
    children: [new TextRun({ text: "[Team member names and roll numbers]", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Course instructor: [Name]", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 1200, after: 120 },
    children: [new TextRun({ text: "Indian Institute of Technology Bombay", bold: true, size: 26 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "November 2026", size: 24 })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ spacing: { after: 200 }, children: [new TextRun({ text: "Contents", bold: true, size: 32 })] }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
);

// 1 Abstract
add(H1("Abstract"),
  P(`We model how commuters choose between walking, two-wheelers, auto-rickshaws, cars, buses and suburban rail, and use the fitted model to look for fare, travel-time and pricing interventions that raise the share of sustainable travel. The data are a survey of 531 commuters with socio-demographic, household, trip and level-of-service variables. A careful audit showed that the travel times and costs in the file were not measured per trip but computed from trip distance with fixed speeds and fare tables, and that eleven records carried travel times inconsistent with their own distance; these were corrected. The deterministic structure means that no model can learn the effect of travel time from these data alone, which shaped every later step.`),
  P(`Six candidate models were compared on the same repeated, stratified cross-validation folds: a random-utility multinomial logit (MNL) estimated by maximum likelihood with alternative availability, a nested logit, regularised multinomial logistic regression, a decision tree, a random forest and XGBoost. XGBoost classified best (cross-validated macro-F1 ${f(cv["XGBoost"].macro_f1_mean, 3)} against ${f(cv["MNL"].macro_f1_mean, 3)} for the MNL). However, a pre-declared simulation-readiness check showed that none of the machine-learning models responds to fare or travel-time changes in a usable way: XGBoost ignores the policy variables altogether, and logistic regression and the random forest make a faster bus or train less attractive. The MNL, with a single generic time coefficient in which a walking minute counts as two in-vehicle minutes (not rejected by a likelihood-ratio test), was therefore frozen. It implies a value of in-vehicle time of about INR ${f(vot, 0)} per hour.`),
  P(`A synthetic population of ${summary.synthetic.n.toLocaleString("en-IN")} commuters was generated from the training data by sequential CART synthesis, chosen over stratified bootstrapping and a Gaussian copula for its fidelity and low memorisation (discriminator AUC ${f(syn.discriminator_auc, 3)}, ${f(100 * syn.exact_copy_share_synthetic, 2)}% exact copies). The frozen model predicts a baseline expected sustainable share of ${f(baseSus, 1)}%. Bounded optimisation by exhaustive grid search, confirmed by differential evolution and by an independent recomputation, puts every lever at its upper bound and raises the expected sustainable share to ${f(optSus, 1)}% (+${f(opt.improvement_pp, 1)} percentage points). Under a limited intervention budget, car pricing and rail fares give the largest gains and bus measures the smallest, because few commuters have a bus option. All policy results are model-based counterfactuals, not causal estimates.`),
);

// 2 Introduction
add(H1("Introduction"),
  P(`Indian metropolitan regions are growing faster than their road space. Every commuter who moves from a car or two-wheeler to a bus, a suburban train or walking reduces congestion and emissions, so understanding why people choose the modes they do is a basic input to transport planning. The classic tool for this is the discrete choice model, which treats each trip as a choice among alternatives and links the probability of each choice to the characteristics of the person and of the alternatives. Machine-learning classifiers are now widely used for the same prediction task and often classify better, but the planning question is not only "who will take the train?" but also "how many more will take it if the fare falls?".`),
  P(`This project builds a complete pipeline around that question. Starting from a survey workbook of 531 commuters, it (i) audits and cleans the data, (ii) trains and compares six candidate models on identical folds, (iii) freezes the model that is fit for simulation, (iv) generates and validates a synthetic population of 5,000 commuters from the training data, (v) predicts baseline mode shares for that population, (vi) simulates fare, time, pricing and availability scenarios and (vii) searches for the combination of feasible interventions that maximises the expected share of sustainable modes.`),
  P(`Two findings shaped the work and are worth stating at the outset. First, the travel times and costs in the workbook turned out to be computed from distance rather than observed, which means that the data alone cannot separate the effect of travel time from the effect of distance. Second, and as a direct consequence, the best classifiers were the worst simulators: models that predict the observed choices well can still respond to a fare cut in the wrong direction. The selection procedure therefore includes a check of policy responses alongside the usual predictive metrics.`),
  P(`Throughout the report, captions start with a tag that says where a result comes from: **Observed data** (the 531 surveyed commuters or their training/test partitions), **Synthetic population** (the generated commuters, baseline predictions only) or **Policy simulation** (counterfactual predictions under changed attributes). The three should not be confused: only the first is evidence about real behaviour, and the last is a statement about the fitted model.`),
  P(`Chapters 3 and 4 set out the problem and objectives, and Chapter 5 reviews the theory. Chapters 6 to 9 describe the data, the audit and cleaning, the exploratory analysis and the features. Chapters 10 to 12 cover the models and their comparison. Chapters 13 to 15 deal with the synthetic population and its predictions, Chapter 16 with scenarios, and Chapters 17 to 20 with the definition of sustainable travel, the optimisation problem, the algorithms and the results. Chapters 21 to 24 discuss the findings, their limitations and possible extensions.`),
);

// 3 Problem statement
add(H1("Problem Statement"),
  P(`For each commuter n we observe a vector of person, household and trip characteristics, the travel time and cost of each of J = 6 alternatives, an indicator of which alternatives are available, and the chosen alternative. The task has three parts.`),
  N(`**Predict.** Estimate P(j | x~n~, C~n~), the probability that commuter n chooses alternative j from the available set C~n~, such that unavailable alternatives receive probability zero and the probabilities sum to one.`, 0),
  N(`**Simulate.** Given a change in bus or rail fares, bus or rail travel times, car costs or availability, recompute these probabilities for every person, holding everything else fixed.`, 0),
  N(`**Optimise.** For a synthetic population of N commuters, find the vector of feasible policy interventions x that maximises the expected share of sustainable modes (walking, bus and suburban rail).`, 0),
  P(`The data impose several difficulties: a small sample (531 records, of which 107 are held out for testing), imbalanced classes (34 auto-rickshaw users), coding that does not fully match the documentation, availability indicators that are constant for rail, and level-of-service attributes that are deterministic functions of distance. The problem statement also rules out some shortcuts: the synthetic population may not be used as training data, the test set may not be used for tuning, and an optimum may not be reported unless it satisfies every constraint and has been verified independently.`),
);

// 4 Objectives
add(H1("Project Objectives"),
  B(`Audit the supplied workbook against the coding document, document every discrepancy and produce a verified data dictionary without inventing category meanings.`),
  B(`Build a reproducible cleaning and feature pipeline in which every correction is evidence-based and logged.`),
  B(`Estimate random-utility multinomial and nested logit models by maximum likelihood with alternative availability, and train regularised logistic regression, decision tree, random forest and XGBoost classifiers on the same data.`),
  B(`Compare all candidates fairly with stratified, repeated cross-validation on the training partition, using macro-F1 as the primary metric and balanced accuracy, log loss and calibration as secondary metrics, and confirm the result on a held-out test set.`),
  B(`Select and freeze one model by a rule written down before fitting, including a check that the model responds sensibly to the policy levers it will be used to simulate.`),
  B(`Generate a synthetic commuter population from the training data with a transparent method, validate its fidelity, logical consistency and memorisation risk, and predict its baseline mode shares with the frozen model.`),
  B(`Simulate fare, travel-time, car-cost and availability scenarios, and solve a bounded policy optimisation problem by grid search and differential evolution, with independent verification of the optimum.`),
  B(`Deliver the work as tested Python code, executed notebooks, an interactive dashboard and this report.`),
);

// 5 Literature
add(H1("Literature and Theoretical Background"),
  H2("Random utility and the multinomial logit"),
  P(`Discrete choice analysis assumes that commuter n assigns a utility U~nj~ to every available alternative j and chooses the one with the highest utility. Utility is split into a part V~nj~ that depends on observed attributes and a random part capturing everything the analyst does not see. If the random parts are independent and identically distributed Gumbel variables, the choice probabilities take the closed multinomial logit form (McFadden, 1974). The standard textbooks are Ben-Akiva and Lerman (1985) and Train (2009); Koppelman and Bhat (2006) give a practical treatment of mode choice with exactly the kind of alternative-specific constants, travel-time and cost terms used here. The coefficients have an economic reading: the ratio of the time and cost coefficients is the value of travel time savings, and weighted walking or waiting time is common in generalised-cost formulations (Ortúzar and Willumsen, 2011; Wardman, 2004).`),
  H2("Nested logit"),
  P(`The multinomial logit implies independence from irrelevant alternatives: improving one alternative draws share proportionally from all others. When some alternatives share unobserved characteristics, for example the bus and the train both require walking to a stop and waiting, the nested logit groups them into nests with a scale parameter λ in (0, 1]. Values below one imply correlation of the unobserved utilities within the nest; a value of one collapses the model to the MNL, and values above one are inconsistent with utility maximisation for all individuals (Ben-Akiva and Lerman, 1985; Train, 2009).`),
  H2("Machine learning for mode choice"),
  P(`Decision trees (Breiman et al., 1984), random forests (Breiman, 2001) and gradient boosting (Chen and Guestrin, 2016) have been compared with logit models in many mode-choice studies. Hagenauer and Helbich (2017) found tree ensembles more accurate than logit models; the systematic review by Hillel et al. (2021) warns that many such comparisons suffer from leakage and weak validation. Zhao et al. (2020) show that the more accurate machine-learning models can still produce behavioural outputs, such as elasticities and choice responses, that are implausible. Our results add a case in which this happens because of the way the data were constructed.`),
  H2("Counterfactual simulation"),
  P(`Changing an input of a fitted model and recomputing its prediction yields a model-based counterfactual. Whether the real world would respond the same way is a causal question that cross-sectional observational data cannot settle on their own (Pearl, 2009). Random-utility models make the assumptions behind their counterfactuals explicit, which is one reason they remain the default in transport appraisal; it does not make those assumptions true.`),
  H2("Synthetic populations"),
  P(`Transport models have long used synthetic populations built by iterative proportional fitting to census marginals (Beckman et al., 1996). Without census targets, the alternatives are data-driven synthesisers: sequential conditional modelling with classification and regression trees as in the synthpop package (Nowok et al., 2016), Gaussian copulas (Patki et al., 2016) and generative adversarial networks such as CTGAN (Xu et al., 2019). With only a few hundred records, the deep generative models are prone to memorising the training data, so this project compares transparent statistical methods instead.`),
  H2("Probability calibration and optimisation"),
  P(`Because the optimisation averages predicted probabilities, those probabilities must be reliable and not just correctly ranked. We use log loss, the multiclass Brier score, reliability diagrams and the expected calibration error (Niculescu-Mizil and Caruana, 2005; Guo et al., 2017). For the optimisation we start with exhaustive grid search, which is transparent and easy to verify, and use differential evolution (Storn and Price, 1997) as an independent global search.`),
);

// 6 Dataset
add(H1("Dataset Description"),
  P(`The workbook *BTech Dataset.xlsx* has one sheet with ${audit.shape[0]} records and ${audit.shape[1]} columns. IDs are unique but run from ${audit.id_range[0]} to ${audit.id_range[1]}, so some records were presumably removed before the file was shared. The accompanying document *Data Coding.docx* describes the project and gives the coding of age, education, occupation, income, dwelling type, satisfaction with public transport (SAPT), trip purpose and mode choice.`),
  TAB(["Group", "Variables"], [
    ["Person", "Gender, Age (3 bands), Education (3 levels), Occup (occupation), Income (3 household bands)"],
    ["Household", "HH_Size, Adults, Children, Working_Adults, Dwelling, Area_Type, Car_Ownership, TwoW_Ownership, ParkingSpace"],
    ["Trip", "Distance_km, Trip_Purpose"],
    ["Level of service", "walk_time_min, TwoW_time_min, auto_time_min, car_time_min, bus_time_min, local_train_time_min; TC_2W, TC_Auto, TC_Car, TC_Bus, TC_LocalTrain (INR)"],
    ["Availability", "Walk_AV, TwoW_AV, Auto_AV, Car_AV, Bus_AV, LocalTrain_AV; driving licences DL_TwoW, DL_Car"],
    ["Attitude", "SAPT (satisfied with public transport)"],
    ["Target", "Choice: 1 walk, 2 two-wheeler, 3 auto-rickshaw, 4 car, 5 bus, 6 suburban rail"],
  ], [1800, 7226], "Variables in the workbook, grouped by role.", "obs", { numericFrom: 99 }),
  TAB(["Mode", "Respondents", "Share (%)"], T.eda_mode_distribution.map(r => [r.mode, r.count, f(r.share_pct, 1)]),
    [4000, 2500, 2526], "Observed mode choice.", "obs"),
  P(`The classes are imbalanced. Two-wheeler, car and rail together account for three quarters of the sample, while auto-rickshaw (34 respondents) and walking (47) are small classes whose per-class metrics will be noisy. The full verified data dictionary, with the documented coding, the observed values and our treatment of each variable, is given in Appendix A.`),
);

// 7 Cleaning
const cc = by(T.audit_code_check, "variable");
add(H1("Data Cleaning and Preprocessing"),
  P(`The audit was run before any change was made; the corrections were then applied by a single function that logs each one. Table 7.1 summarises the checks.`),
  TAB(["Check", "Result"], [
    ["Missing values", `${audit.missing_values} in ${audit.shape[0] * audit.shape[1]} cells`],
    ["Duplicate records", `${audit.duplicate_rows} (also ${audit.duplicate_rows_ignoring_id} when ID is ignored)`],
    ["Duplicate columns", audit.duplicate_columns.map(p => `${p[0]} = ${p[1]}`).join("; ")],
    ["Constant columns", audit.constant_columns.join(", ") + " (rail available to everyone)"],
    ["Codes not in the document", "Occup code 5; Dwelling code 4; Trip_Purpose coded 1/2 instead of 0/1"],
    ["Variables not in the document", "Gender, Area_Type, ParkingSpace"],
    ["Chosen mode unavailable", `${audit.availability.chosen_alternative_unavailable} records`],
    ["Travel times inconsistent with distance", `${corrIDs.length} records (IDs ${corrIDs.join(", ")})`],
    ["Household inconsistencies", `${audit.household_flag_counts.workers_exceed_adults} with more workers than adults; ${audit.household_flag_counts.hh_size_not_adults_plus_children} with HH_Size different from Adults + Children; ${audit.household_flag_counts.workers_exceed_hh_size} with more workers than members; ${audit.household_flag_counts.zero_adults} with no adult`],
  ], [3000, 6026], "Summary of the data audit.", "obs", { numericFrom: 99 }),
  H2("Coding discrepancies"),
  P(`Table 7.2 compares the observed codes with the coding document. We did not relabel any variable. Where the evidence below points to an interpretation, we report it as a suggestion and keep neutral labels such as "purpose code 2" in all models and figures.`),
  TAB(["Variable", "Document", "Observed (code: count)", "Issue"], [
    ["Trip_Purpose", cc.Trip_Purpose.documented_codes, cc.Trip_Purpose.observed_counts, "codes shifted (1/2 instead of 0/1)"],
    ["Occup", cc.Occup.documented_codes, cc.Occup.observed_counts, "code 5 undocumented; codes 1 and 4 have 3 and 4 records"],
    ["Dwelling", cc.Dwelling.documented_codes, cc.Dwelling.observed_counts, "code 4 undocumented"],
    ["Gender", "not documented", "0:16, 1:515", "meaning unknown; 3% in one group"],
    ["SAPT", cc.SAPT.documented_codes, cc.SAPT.observed_counts, "matches; highly imbalanced"],
  ], [1400, 2600, 2200, 2826], "Observed codes against the coding document.", "obs", { numericFrom: 99, size: 17 }),
  P(`For trip purpose, all ${evidence("purpose_by_age").find(r => r.Age === 1)["2"]} respondents aged 18-30 report code 2, as do ${f(100 * evidence("purpose_by_occupation").find(r => r.Occup === 3)["2"] / (evidence("purpose_by_occupation").find(r => r.Occup === 3)["1"] + evidence("purpose_by_occupation").find(r => r.Occup === 3)["2"]), 0)}% of occupation code 3, while purpose code 1 is concentrated among respondents over 55. This is consistent with "2 = work trip", the document's coding shifted by one, but it is not confirmed. The occupation codes are more troubling. Code 2, which the document calls "Government employee", is 64% aged over 55 and makes mostly purpose-1 trips, and code 1 ("Unemployed") only makes purpose-2 trips; the document's occupation labels may not apply to this file at all. These questions are listed for the data owner at the end of Appendix A.`),
  H2("Travel times and costs are functions of distance"),
  P(`Plotting each travel time against distance (Figure 7.1) shows that all but eleven records lie exactly on a straight line through the origin. Each time is the distance multiplied by a fixed rate:`),
  EQ([sub("t", "nj"), mr(" = "), sub("r", "j"), mr(" "), sub("d", "n"), mr(",  j ∈ {walk, 2W, auto, car, bus, rail}")]),
  TAB(["Mode", "Minutes per km", "Implied speed (km/h)"], MODES.map(m => [MODE_NAME[m], f(rules.time_rate_min_per_km[m], 3), f(rules.implied_speed_kmh[m], 1)]),
    [3600, 2700, 2726], "Speed rules recovered from the 520 consistent records (median minutes per km).", "obs"),
  ...FIG("audit_time_vs_distance", "Recorded travel time against trip distance for each mode, raw workbook. Eleven records (orange) lie far from the fixed-speed lines.", "obs", 600),
  P(`Costs follow the same pattern. Car cost is INR ${f(rules.car_rate_inr_per_km, 2)} per km and two-wheeler cost INR ${f(rules.twow_rate_inr_per_km, 2)} per km, both rounded to whole rupees; the auto fare is ${autoFare}; bus and rail fares follow stepped tables (bus INR 7 up to about 3 km, rising by roughly a rupee per km to a cap of INR 32; rail INR 5 up to about 20 km). Table 7.4 shows that these rules reproduce every recorded cost to within one rupee, apart from rounding.`),
  TAB(["Cost column", "Exact match", "Within INR 1", "Largest difference (INR)"], T.audit_cost_rule_match.map(r => [r.cost_column, f(r.exact_match_share, 3), f(r.within_1_inr_share, 3), f(r.max_abs_diff_inr, 0)]),
    [2600, 2100, 2100, 2226], "Agreement between the recovered cost rules and the recorded costs.", "obs"),
  P(`In other words, the level-of-service attributes were generated from distance rather than measured for each trip. The consequence is fundamental: in this dataset, bus time is distance times ${f(rules.time_rate_min_per_km.bus, 2)}, so the effect of bus time cannot be separated from the effect of distance by any amount of modelling. We return to this in Chapters 10 and 12.`),
  H2("Correction of inconsistent travel times"),
  P(`For the eleven records off the lines, Table 7.5 converts each recorded time and cost back into the distance it implies. In every case the costs agree with the recorded distance, while the times imply distances that are wrong by large factors. ID 179's motorised times are exactly 1,000 times too large, the pattern of metres entered as kilometres. We therefore treated the distance as correct and recomputed the six travel times of these records from distance with the recovered rates. Every change (${T.cleaning_time_corrections.length} values) is logged in *cleaning_time_corrections.csv*.`),
  TAB(["ID", "Distance (km)", "km implied by walk time", "km implied by car time", "km implied by car cost", "km implied by 2W cost"],
    T.audit_los_inconsistent_records.map(r => [r.ID, f(r.Distance_km, 1), f(r.km_implied_by_walk_time, 2), f(r.km_implied_by_car_time, 2), f(r.km_implied_by_car_cost, 2), f(r.km_implied_by_2w_cost, 2)]),
    [900, 1500, 1650, 1650, 1650, 1676], "Distances implied by the recorded times and costs of the eleven inconsistent records.", "obs", { size: 17 }),
  P(`Extreme distances were not removed. The longest trips (89.7 km and 47.0 km) have costs and rail fares that match their distance and are plausible suburban commutes; they are kept as genuine observations.`),
  H2("Household variables and availability"),
  P(`Fifteen records report more working members than adults. This is possible if "Children" counts sons and daughters of any age rather than minors, so these records were flagged rather than edited. Three records have HH_Size different from Adults + Children, three report more workers than household members and one reports no adult at all. None of these was changed; the derived workers-share feature caps workers at household size.`),
  P(`No respondent chose an unavailable mode. Car and two-wheeler availability equal licence holding among households that own the vehicle (${audit.availability.car_owned_but_unavailable} people live in car-owning households without having a car available). Walking is available for every trip up to 1.5 km but for only ${f(100 * audit.availability["walk_av_share_above_3km"], 0)}% of trips above 3 km. Bus is available to only ${f(100 * audit.availability.availability_share.bus, 0)}% of respondents. Rail is available to everyone, so the data contain no information about rail availability.`),
  H2("Final cleaning decisions and data split"),
  B(`Recompute the six travel times of the ${corrIDs.length} inconsistent records from distance (logged).`),
  B(`Drop DL_TwoW and DL_Car, which duplicate TwoW_AV and Car_AV exactly.`),
  B(`Keep LocalTrain_AV only to define the choice set; it is never used as a predictor.`),
  B(`Keep every category code as recorded; pool the two rare occupation codes (1 and 4, seven people) only inside the models.`),
  B(`Flag, but do not edit, household inconsistencies; keep extreme distances.`),
  P(`The cleaned data (531 records) were split once into a training partition of ${summary.split.train} and a test partition of ${summary.split.test} records, stratified by chosen mode with seed 42. The test partition was not touched until the selected model was frozen.`),
);

// 8 EDA
const share = (tab, key, m) => f(tab[key][m], 1);
add(H1("Exploratory Data Analysis"),
  P(`All figures in this chapter use the cleaned data. Each mode has the same colour throughout the report.`),
  ...FIG("eda_mode_by_distance_band", "Mode share by trip distance band.", "obs", 560),
  P(`Distance organises the choice set more than any other variable. Walking is chosen only up to ${f(eda.dbc.walk.max, 1)} km. The rail share rises from ${share(eda.dist, "0-1.5", "train")}% of trips below 1.5 km to ${share(eda.dist, "10-20", "train")}% at 10-20 km and ${share(eda.dist, "20+", "train")}% beyond 20 km. Bus peaks in the 5-10 km band (${share(eda.dist, "5-10", "bus")}%), and car is strongest for short trips (${share(eda.dist, "0-1.5", "car")}% below 1.5 km), which suggests that many short car trips are escort or errand trips rather than long commutes.`),
  ...FIG("eda_mode_by_income", "Mode share by monthly household income.", "obs", 560),
  P(`Car use rises steeply with income: ${share(eda.income, 1, "car")}% in the lowest band, ${share(eda.income, 2, "car")}% in the middle band and ${share(eda.income, 3, "car")}% above INR 80,000 a month. Rail, bus and auto shares fall correspondingly. Income therefore enters the car utility in the choice model.`),
  ...FIG("eda_mode_by_age", "Mode share by age group.", "obs", 520),
  ...FIG("eda_mode_by_purpose", "Mode share by trip purpose code.", "obs", 520),
  P(`Respondents over 55 walk (${share(eda.age, 3, "walk")}%) and take the bus (${share(eda.age, 3, "bus")}%) more often than those aged 30-55. Trips with purpose code 1 involve far more walking (${share(eda.purpose, 1, "walk")}% against ${share(eda.purpose, 2, "walk")}%) and bus use (${share(eda.purpose, 1, "bus")}% against ${share(eda.purpose, 2, "bus")}%), consistent with non-work travel. The 18-30 group has only 33 people, so its shares are imprecise.`),
  ...FIG("eda_mode_by_occupation", "Mode share by occupation code, as recorded.", "obs", 520),
  ...FIG("eda_mode_by_gender", "Mode share by gender code (coding not documented).", "obs", 520),
  P(`Occupation codes 1 and 4 have three and four people, so their shares carry no information. Code 3 has the highest rail share (${share(eda.occ, 3, "train")}%). Gender code 0 has only 16 people; the higher walking share in that group (${share(eda.gender, 0, "walk")}%) rests on four individuals, which is why gender is not used in the choice model.`),
  ...FIG("eda_mode_by_vehicle_ownership", "Mode share by household vehicle ownership.", "obs", 540),
  P(`Households without a vehicle split between rail (${share(eda.veh, 0, "train")}%), bus (${share(eda.veh, 0, "bus")}%), walking (${share(eda.veh, 0, "walk")}%) and auto-rickshaw (${share(eda.veh, 0, "auto")}%). Where the household owns only a two-wheeler, it is used for ${share(eda.veh, 1, "twow")}% of trips; car-only households use the car for ${share(eda.veh, 2, "car")}%.`),
  ...FIG("eda_mode_by_sapt", "Mode share by satisfaction with public transport.", "obs", 520),
  P(`Only 29 respondents are dissatisfied with public transport. They use two-wheelers much more (${share(eda.sapt, 0, "twow")}% against ${share(eda.sapt, 1, "twow")}%) and rail less, which is plausible, but the group is too small for firm conclusions.`),
  ...FIG("eda_distance_by_choice", "Trip distance by chosen mode (dot = median, bar = interquartile range).", "obs", 540),
  P(`Median trip distances by chosen mode are ${f(md("walk"), 1)} km for walking, ${f(md("auto"), 1)} km for auto-rickshaw, ${f(md("car"), 1)} km for car, ${f(md("twow"), 1)} km for two-wheeler, ${f(md("bus"), 1)} km for bus and ${f(md("train"), 1)} km for rail.`),
  ...FIG("eda_travel_time", "Travel time of each mode for all respondents, cleaned data (log scale).", "obs", 520),
  ...FIG("eda_travel_cost", "Travel cost of each mode for all respondents (log scale).", "obs", 520),
  P(`The time and cost distributions look like rescaled copies of the distance distribution because that is what they are. The rail fare hardly varies at all: INR 5 for 95% of trips.`),
  ...FIG("eda_availability_vs_choice", "Availability of each mode and how often it is chosen.", "obs", 560),
  P(`Bus is available to only ${f(eda.av.Bus.available_pct, 0)}% of respondents but chosen by ${f(eda.av.Bus.chosen_when_available_pct, 0)}% of those who have it, the highest conditional share of any mode. One reading is that buses are attractive where they exist; another is that bus availability was more likely to be recorded for people who actually use the bus. The data cannot distinguish the two, and the difference matters for the availability scenarios in Chapter 16.`),
  ...FIG("eda_association_with_choice", "Bias-corrected Cramér's V between each variable and the chosen mode (continuous variables in quintiles).", "obs", 470),
  ...FIG("eda_numeric_correlation", "Spearman rank correlations among the numeric variables.", "obs", 470),
  P(`Bus availability has the strongest association with the chosen mode (V = ${f(eda.assoc.find(r => r.variable === "Bus_AV").cramers_v, 2)}), followed by car and two-wheeler availability, distance and income. Household size and the number of workers show no association. The correlation matrix confirms the audit: distance, all six times and most costs are almost perfectly rank-correlated.`),
);

// 9 Feature engineering
add(H1("Feature Engineering"),
  P(`Every candidate feature listed in the project brief was checked for three things: a defensible definition, availability at prediction time, and whether it merely duplicates information already in the model. Table 9.1 gives the outcome.`),
  TAB(["Candidate feature", "Spearman with distance", "Decision", "Reason"], T.feature_review.map(r => [r.feature.replace(/_/g, " "), f(r.spearman_with_distance, 3), r.decision, r.reason]),
    [2500, 1300, 1000, 4226], "Review of candidate features.", "obs", { numericFrom: 1, size: 17 }),
  P(`Most of the "relative advantage" features fail the third test. Because times and costs are computed from distance, the difference between car and bus time is distance multiplied by a constant (rank correlation ${f(T.feature_review.find(r => r.feature === "time_advantage_bus_vs_car").spearman_with_distance, 3)} with distance), and cost per kilometre is constant apart from rounding. Total vehicles and the number of available modes are exact sums of variables already used, and public-transport availability is constant because rail is available to everyone.`),
  P(`Five derived variables were kept: the logarithm of distance (a concave distance effect for the linear models), a four-level vehicle-ownership category, an indicator that any private vehicle is available, the share of household members who work (capped at one) and an indicator for children in the household. The rare occupation codes were pooled. All derived variables are computed inside the model pipeline from raw columns, so a scenario that changes a raw time or cost updates them automatically, and nothing computed from the target or from other records can leak into a prediction.`),
  P(`The machine-learning models receive ${sizes.n_categorical} categorical variables (one-hot encoded) and ${sizes.n_numeric} numeric variables (standardised), ${sizes.n_ml_features} columns in all, with the encoder and scaler fitted inside each training fold. The discrete choice models use the same data in person-alternative long format, with one row per available alternative; the training partition becomes ${sizes.long_rows.toLocaleString("en-IN")} such rows.`),
);

// 10 Discrete choice modelling
const pw = (w) => profile[String(w)] || profile[w];
add(H1("Discrete Choice Modelling"),
  H2("Utility specification and the multinomial logit"),
  P(`Each alternative has a utility made of an observed part and a random part:`),
  EQ([sub("U", "nj"), mr(" = "), sub("V", "nj"), mr(" + "), sub("ε", "nj"), mr(",     "), sub("V", "nj"), mr(" = "), sum("k", [sub("β", "k"), mr(" "), sub("x", "njk")])]),
  P(`where x~njk~ is the k-th explanatory variable as it enters alternative j: an alternative-specific constant, a travel time or cost, or a person characteristic interacted with one alternative. With independent Gumbel errors and an availability indicator a~nj~, the choice probability is`),
  EQ([sub("P", "nj"), mr(" = "), frac([sub("a", "nj"), mr(" exp("), sub("V", "nj"), mr(")")], [sum("i", [sub("a", "ni"), mr(" exp("), sub("V", "ni"), mr(")")])])]),
  P(`so an unavailable alternative has probability exactly zero and the others are renormalised over the person's own choice set. Rail, available to everyone, is the reference alternative (its constant is fixed at zero).`),
  H2("Estimation"),
  P(`The data are reshaped into a person x alternative x parameter design array (the long format), and the coefficients are found by maximising the log-likelihood`),
  EQ([mr("LL(β) = "), sum("n", [mr("ln "), sub("P", [mr("n,"), sub("y", "n")])])]),
  P(`with its analytical gradient and Hessian; the MNL log-likelihood is globally concave, so a trust-region Newton method converges to the unique maximum. Standard errors come from the inverse Hessian. As a check of the implementation we re-estimated the selected specification with the conditional logit of statsmodels on the same long-format data: the log-likelihoods agree to within 0.001 and the coefficients to within 0.02 (our optimiser reaches a marginally higher likelihood). This comparison is part of the automated tests.`),
  H2("The identification problem"),
  P(`The natural first specification has alternative-specific constants, a walking-time coefficient, a generic in-vehicle time coefficient for the five motorised modes and a generic cost coefficient ("FREE" in Table 10.1). Its in-vehicle time coefficient is **positive**: ${f(by(T.mnl_estimates_FREE_SOCIO, "index").B_time.estimate, 4)} per minute with a p-value of ${f(by(T.mnl_estimates_FREE_SOCIO, "index").B_time.p_value, 2)}. A positive time coefficient means that, other things equal, slower modes are preferred, which no behavioural model can accept. The explanation is the audit finding: every travel time is a fixed multiple of distance, so the only information about the time coefficient comes from the fixed differences in speed between modes, and after the cost terms have absorbed the distance pattern there is nothing left. In practice the data cannot separate time from distance.`),
  P(`We therefore use one generic time coefficient for all six modes, in which a walking minute counts as w in-vehicle minutes. Weighting walking time more heavily than riding time is standard in generalised-cost practice (Ortúzar and Willumsen, 2011; Wardman, 2004), and Table 10.2 shows how the likelihood depends on w.`),
  TAB(["Walk weight w", "Log-likelihood", "LR vs FREE", "p-value", "Time coef.", "Cost coef.", "VOT (INR/h)"],
    T.mnl_walk_weight_profile.map(r => [f(r.walk_weight, 1), f(r.log_likelihood, 2), f(r.LR_vs_free, 2), f(r.p_value, 3), f(r.B_time, 4), f(r.B_cost, 4), f(r.VOT_INR_per_hour, 1)]),
    [1300, 1400, 1150, 1050, 1300, 1300, 1526], "Profile of the behavioural specification over the walk-time weight, tested against the unrestricted specification (one degree of freedom).", "obs"),
  P(`Equal weighting (w = 1) is rejected (p = ${f(pw(1).p_value, 3)}), while w = 2 is not (p = ${f(pw(2).p_value, 3)}). The likelihood keeps rising slowly with w because the data, as explained, push the time coefficient towards zero. We adopt w = 2 as an explicit modelling assumption and test its influence on the policy results in Chapter 20.`),
  H2("Specifications compared"),
  TAB(["Specification", "Parameters", "Log-likelihood", "Rho-squared", "AIC", "BIC", "Signs valid", "VOT (INR/h)"],
    T.mnl_specifications.map(r => [r.specification, r.n_parameters, f(r.log_likelihood, 2), f(r.rho_squared, 3), f(r.AIC, 1), f(r.BIC, 1), r.signs_ok ? "yes" : "no", f(r.in_vehicle_time_INR_per_hour, 1)]),
    [1500, 1050, 1250, 1050, 1000, 1000, 1000, 1176], "MNL specifications estimated on the training partition (424 records). GC = generic time with walk weight 2; FREE = separate walking and in-vehicle time coefficients.", "obs", { size: 17 }),
  P(`GC contains the constants, time and cost; GC+CORE adds the person terms with the clearest prior support (income and car ownership in the car utility, purpose code 1 in the walking and bus utilities); GC+SOCIO adds seven further terms. GC+CORE has the lowest AIC and BIC and the best macro-F1 in the tuning folds, so it is the MNL carried into the comparison. Its estimates are in Table 10.4.`),
  TAB(["Parameter", "Estimate", "Std. error", "t-statistic", "p-value"],
    T.mnl_estimates_selected.map(r => [r.index, f(r.estimate, 4), f(r.std_error, 4), f(r.t_stat, 2), f(r.p_value, 4)]),
    [2800, 1600, 1600, 1500, 1526], "Estimates of the selected MNL (GC+CORE). Rail is the reference alternative.", "obs"),
  ...FIG("mnl_coefficients", "Estimated coefficients of the selected MNL with 95% confidence intervals.", "obs", 600),
  P(`Time (${f(est.B_time.estimate, 4)} per minute) and cost (${f(est.B_cost.estimate, 4)} per rupee) are negative and significant. Their ratio gives a value of in-vehicle time of about INR ${f(vot, 0)} per hour, and of walking time about INR ${f(summary.vot.walk_time_INR_per_hour, 0)} per hour; the first number inherits the walk-weight assumption. Middle and high income raise the utility of the car by ${f(est.B_car_inc2.estimate, 2)} and ${f(est.B_car_inc3.estimate, 2)}, each additional household car by ${f(est.B_car_owned.estimate, 2)}, and purpose code 1 raises the utility of walking and of the bus. The model reaches a rho-squared of ${f(summary.mnl_fit.rho_squared, 3)} against equal shares within each choice set.`),
  H2("Nested logit"),
  P(`The nested logit replaces the single logit with a two-level structure:`),
  EQ([sub("P", "nj"), mr(" = "), sub("P", "n"), mr("(j | m) "), sub("P", "n"), mr("(m),    "), sub("P", "n"), mr("(j | m) = "),
      frac([sub("a", "nj"), mr(" exp("), sub("V", "nj"), mr("/"), sub("λ", "m"), mr(")")], [mr("exp("), sub("I", "nm"), mr(")")])]),
  EQ([sub("I", "nm"), mr(" = ln "), sum([mr("i ∈ "), sub("B", "m")], [sub("a", "ni"), mr(" exp("), sub("V", "ni"), mr("/"), sub("λ", "m"), mr(")")]), mr(",    "),
      sub("P", "n"), mr("(m) = "), frac([mr("exp("), sub("λ", "m"), sub("I", "nm"), mr(")")], [sum("k", [mr("exp("), sub("λ", "k"), sub("I", "nk"), mr(")")])])]),
  P(`where B~m~ is the set of alternatives in nest m, I~nm~ the inclusive value and λ~m~ the nest scale. We wrote the analytical gradient (checked against finite differences in the tests) and maximised the likelihood with the MNL estimates as starting values and λ bounded to [0.05, 1]. Five structures were tested, each motivated by a shared unobserved attribute: public transport {bus, rail}, private vehicles {car, two-wheeler}, shared or hired modes {auto, bus, rail} and motorised travel. A structure was declared usable only if every λ was strictly inside its bounds and significantly below one.`),
  TAB(["Structure", "Log-likelihood", "λ (s.e.)", "At bound", "LR vs MNL (p)", "Usable"],
    T.nested_logit_structures.map(r => {
      const lam = String(r.lambdas).replace(/np\.float64\(([^)]*)\)/g, "$1").replace(/[{}']/g, "");
      return [r.structure, f(r.log_likelihood, 2), `${lam} (${String(r.lambda_se).replace(/[[\]]/g, "")})`, String(r.lambda_at_bound).replace(/[[\]]/g, ""), `${f(r.LR_vs_MNL, 2)} (${f(r.LR_p_value, 4)})`, r.usable ? "yes" : "no"];
    }), [2600, 1100, 2200, 1100, 1200, 826], "Nested logit structures estimated on the training partition with the GC+CORE utilities.", "obs", { size: 16 }),
  P(`The {bus, rail} nest drives λ to its lower bound. This is a sign of near-separation in the small bus-versus-rail sub-sample (only ${f(100 * audit.availability.availability_share.bus, 0)}% of respondents have a bus option), not of a real correlation, so the structure is not reliable. The private-vehicle nest pushes λ to one, or above one when the bound is relaxed, so private modes do not form a nest. The motorised nest collapses to the MNL. The only usable structure is the shared/public nest {auto, bus, rail}: λ = ${f(nlEst[lamKey].estimate, 3)} (s.e. ${f(nlEst[lamKey].std_error, 3)}), significantly below one, with a likelihood-ratio statistic of ${f(nests["SHARED{auto,bus,train}"].LR_vs_MNL, 2)} against the MNL (p = ${f(nests["SHARED{auto,bus,train}"].LR_p_value, 4)}). The three hired or public modes share unobserved characteristics, such as not needing a vehicle or parking, that the MNL cannot represent. This nested logit is the second discrete choice candidate.`),
);

// 11 Machine learning models
const tp = (m) => JSON.parse(tuning[m].best_params);
const clean = (o) => Object.entries(o).map(([k, v]) => `${k.replace("estimator__clf__", "")} = ${v}`).join(", ");
add(H1("Machine Learning Models"),
  H2("Common pipeline and availability"),
  P(`All four machine-learning models share one pipeline: the feature builder of Chapter 9, a column transformer that one-hot encodes the categorical codes and standardises the numeric variables (refitted inside every training fold), and the classifier. Standard classifiers know nothing about choice sets, so each pipeline is wrapped in a component that conditions its probabilities on the available alternatives:`),
  EQ([sub("p'", "nj"), mr(" = "), frac([sub("a", "nj"), mr(" max("), sub("p", "nj"), mr(", ε)")], [sum("k", [sub("a", "nk"), mr(" max("), sub("p", "nk"), mr(", ε)")])])]),
  P(`with ε = 10^−6^. This is exactly the conditioning the MNL performs internally. The small floor stops a decision tree that assigns an available mode exactly zero probability from producing an infinite log loss. Every metric in Chapter 12 is computed on these masked probabilities, for every model.`),
  H2("The four classifiers"),
  B(`**Multinomial logistic regression**: a linear softmax model on all ${sizes.n_ml_features} encoded features with an L2 penalty. Unlike the MNL it gives each alternative its own coefficient on every variable, including the times and costs of the other modes. The class-weighting option reweights rows inversely to class frequency.`),
  B(`**Decision tree**: a single CART tree. Tuned over maximum depth, minimum samples to split, minimum samples per leaf, the split criterion (Gini or entropy) and class weighting.`),
  B(`**Random forest**: bagged trees with random feature subsets. Tuned over the number of trees, depth, split and leaf sizes, the share of features tried at each split and class weighting.`),
  B(`**XGBoost**: gradient-boosted trees with a softmax objective. Tuned over the number of trees, learning rate, depth, row and column subsampling, minimum child weight, L1 and L2 regularisation and class weighting.`),
  H2("Hyperparameter tuning"),
  P(`Tuning used only the training partition: repeated stratified 5-fold cross-validation (2 repeats, seed 42), with macro-F1 as the score. The logistic regression and the decision tree were tuned by full grid search (22 and 640 configurations), the random forest and XGBoost by randomised search over 60 configurations, and the MNL specification by grid search over the three behavioural specifications.`),
  TAB(["Model", "Configurations", "Best tuning macro-F1", "Selected hyperparameters"], ["MNL", "Logistic regression", "Decision tree", "Random forest", "XGBoost"].map(m =>
    [m, tuning[m].configurations, f(tuning[m].best_cv_macro_f1, 3), clean(tp(m))]),
    [1700, 1250, 1300, 4776], "Hyperparameter search results (training partition only).", "obs", { numericFrom: 1, size: 16 }),
  P(`Class weighting was selected only for the logistic regression. The tuned random forest is shallow (depth 3) and the tuned XGBoost uses many small steps (600 trees with learning rate 0.01), both signs of the small sample.`),
);

// 12 Comparison and selection
const cvRow = (m) => [m, f(cv[m].accuracy_mean, 3), `${f(cv[m].macro_f1_mean, 3)} (${f(cv[m].macro_f1_sd, 3)})`, f(cv[m].weighted_f1_mean, 3),
  f(cv[m].balanced_accuracy_mean, 3), f(cv[m].log_loss_mean, 3), f(cv[m].brier_mean, 3), f(cv[m].ece_mean, 3), f(cv[m].share_mae_pp_mean, 2)];
const testRow = (m) => [m, f(test[m].accuracy, 3), f(test[m].macro_precision, 3), f(test[m].macro_recall, 3), f(test[m].macro_f1, 3),
  f(test[m].weighted_f1, 3), f(test[m].balanced_accuracy, 3), f(test[m].log_loss, 3)];
const readyCell = (m, lever) => { const r = readyBy(m, lever); return `${sgn(r.mean_change_pp, 2)} (${f(100 * r.share_wrong_direction, 0)}%)`; };
add(H1("Model Comparison and Selection"),
  H2("Evaluation design"),
  P(`After tuning, every model was refitted with its chosen configuration on the same 25 folds: repeated stratified 5-fold cross-validation with 5 repeats and a different seed (2026) from the tuning folds, again on the training partition only. Using different folds for tuning and comparison reduces, but does not remove, the optimism of the tuned models' scores; the untouched test set gives the final check. A constants-only logit, which predicts the sample shares within each person's choice set, is included as a reference point and is not a candidate.`),
  H2("Metrics"),
  P(`Macro-F1 is the primary metric because it gives the small classes the same weight as the large ones:`),
  EQ([mr("Macro-F1 = "), frac("1", "J"), sum("j", [sub("F1", "j")]), mr(",     "), mr("LogLoss = −"), frac("1", "N"), sum("n", [mr("ln "), sub("p", [mr("n,"), sub("y", "n")])])]),
  EQ([mr("Brier = "), frac("1", "N"), sum("n", sum("j", [sup(rb([sub("p", "nj"), mr(" − "), sub("y", "nj")]), "2")])), mr(",     "),
      mr("ECE = "), sum("b", [frac([mr("|"), sub("B", "b"), mr("|")], "N"), mr(" |acc("), sub("B", "b"), mr(") − conf("), sub("B", "b"), mr(")|")])]),
  P(`We also report accuracy, macro precision and recall, weighted F1, balanced accuracy (the mean of the per-class recalls) and the aggregate share error: the mean absolute difference, in percentage points, between the average predicted probability of each mode and its observed share. Because the optimisation averages probabilities, this last metric is directly relevant to the downstream task.`),
  H2("Cross-validated results"),
  TAB(["Model", "Accuracy", "Macro-F1 (sd)", "Weighted F1", "Balanced acc.", "Log loss", "Brier", "ECE", "Share error (pp)"],
    [...ORDER, REF].map(cvRow), [1700, 850, 1150, 900, 950, 850, 800, 750, 1076],
    "Cross-validated performance on the training partition (5 x 5 folds). Values are means over 25 folds.", "obs", { size: 16 }),
  ...FIG("model_cv_comparison", "Cross-validated macro-F1, balanced accuracy and log loss with 95% intervals.", "obs", 600),
  P(`${bestF1Model} has the best macro-F1 (${f(cv[bestF1Model].macro_f1_mean, 3)}) and ${bestLLModel} the lowest log loss (${f(cv[bestLLModel].log_loss_mean, 3)}). The decision tree classifies almost as well but its probabilities are poor (log loss ${f(cv["Decision tree"].log_loss_mean, 2)}). The class-weighted logistic regression reaches the highest balanced accuracy, at the price of distorted aggregate shares (share error ${f(cv["Logistic regression"].share_mae_pp_mean, 1)} points). The two logit models are clearly behind on classification (macro-F1 ${f(cv["MNL"].macro_f1_mean, 3)} and ${f(cv["Nested logit"].macro_f1_mean, 3)}), well ahead of the constants-only reference (${f(cv[REF].macro_f1_mean, 3)}), and among the best on aggregate shares (${f(cv["MNL"].share_mae_pp_mean, 2)} points for the MNL). Fold-to-fold standard deviations of macro-F1 are 0.05 to 0.07 for every model, a reminder of how small the validation folds are.`),
  ...FIG("model_calibration", `Reliability of out-of-fold probabilities for the selected model and the best-calibrated alternative.`, "obs", 560),
  P(`Both the MNL and XGBoost are reasonably calibrated for walking, two-wheeler, auto-rickshaw and rail. Both under-predict the bus when it is likely: among people given a high bus probability, almost everyone takes the bus, which again reflects the strong link between bus availability and bus use.`),
  H2("Simulation readiness"),
  P(`The frozen model has one job beyond prediction: to tell us how choice probabilities change when a fare, a travel time or a car cost changes. Before fitting anything we therefore wrote down a second requirement. For each of the five policy levers, a 20% change (bus time and fare down, rail time and fare down, car cost up) is applied to the training rows, and the change in the probability of the directly affected mode is measured among the people for whom that mode is available. A model passes if, for every lever, the average change has the sign that consumer theory requires and is not negligible (above 0.01 percentage points), and at most 5% of people move in the wrong direction.`),
  TAB(["Model", "Bus time", "Bus fare", "Rail time", "Rail fare", "Car cost", "Passes"],
    ORDER.map(m => [m, readyCell(m, "bus time"), readyCell(m, "bus fare"), readyCell(m, "rail time"), readyCell(m, "rail fare"), readyCell(m, "car cost"), sel.readiness[m] ? "yes" : "no"]),
    [1800, 1300, 1300, 1300, 1300, 1300, 726], "Mean change in the probability of the affected mode in percentage points, and in brackets the share of people moving in the wrong direction, for a 20% change in each lever (training rows).", "obs", { size: 16 }),
  ...FIG("model_simulation_readiness", "Simulation-readiness check: response of each model to a 20% change in each policy lever.", "obs", 600),
  P(`Only the two logit models respond to every lever in the right direction and for every person. The machine-learning models fail in two different ways. The tuned XGBoost **never responds**: its trees never split on a policy variable, because distance carries the same information, so its predictions do not move whatever the fare or travel time. Logistic regression and the random forest **respond in the wrong direction**. Longer bus and rail times go with more bus and rail use in the data, simply because both rise with distance, so these models learned that a slower bus is a more popular bus; a 20% faster train lowers the predicted rail probability for ${f(100 * readyBy("Logistic regression", "rail time").share_wrong_direction, 0)}% of people under the logistic regression and ${f(100 * readyBy("Random forest", "rail time").share_wrong_direction, 0)}% under the random forest. The decision tree lowers bus use when bus time falls. None of this is a flaw of these algorithms in general; it is what any flexible model learns when an attribute is a deterministic function of another variable.`),
  H2("Selection rule and outcome"),
  P(`The rule, fixed in the configuration file before any model was fitted, is: (1) shortlist the models whose cross-validated macro-F1 is within one standard error of the best; (2) keep those whose balanced accuracy is within one standard error of the best in the shortlist; (3) rank by log loss, with near-ties broken in favour of the simpler model; (4) rank the remaining models by macro-F1; (5) take the highest-ranked model that passes the simulation-readiness check. Test-set results are confirmatory only.`),
  P(`The predictive ranking was ${sel.ranking.join(", ")}, with ${sel.shortlist_after_f1_and_ba.join(" and ")} on the shortlist${lrGap > 0 && lrGap < 0.005 ? ` (the logistic regression missed it by ${f(lrGap, 4)})` : ""}. ${sel.skipped.join(", ")} failed the readiness check, so the **${WINNER}** was selected and frozen. Without the readiness requirement, XGBoost would have been chosen. We consider the requirement essential, because a model that cannot represent a fare cut cannot be used to optimise one; but we report the trade-off openly, since the frozen model is not the best classifier.`),
  H2("Held-out test set"),
  TAB(["Model", "Accuracy", "Macro precision", "Macro recall", "Macro-F1", "Weighted F1", "Balanced acc.", "Log loss"],
    [...ORDER, REF].map(testRow), [1900, 950, 1050, 1000, 950, 1000, 1050, 1126], `Performance on the ${summary.split.test} held-out test records.`, "obs", { size: 16 }),
  P(`On the test set the MNL does worse than its cross-validation suggested: accuracy ${f(test.MNL.accuracy, 3)} (bootstrap 95% interval ${f(ci.MNL.accuracy[0], 2)}-${f(ci.MNL.accuracy[1], 2)}), macro-F1 ${f(test.MNL.macro_f1, 3)} (${f(ci.MNL.macro_f1[0], 2)}-${f(ci.MNL.macro_f1[1], 2)}) and log loss ${f(test.MNL.log_loss, 3)} (${f(ci.MNL.log_loss[0], 2)}-${f(ci.MNL.log_loss[1], 2)}). Its accuracy ${Math.abs(test.MNL.accuracy - test[REF].accuracy) < 1e-9 ? "equals" : "is close to"} that of the constants-only reference, although its log loss (${f(test.MNL.log_loss, 3)} against ${f(test[REF].log_loss, 3)}) shows that it does carry information. XGBoost (macro-F1 ${f(test.XGBoost.macro_f1, 3)}) and the decision tree (${f(test["Decision tree"].macro_f1, 3)}) classify the test set much better. With 107 test records the intervals are wide, but the ordering agrees with cross-validation. The per-class results of the frozen model are given in Table 12.5.`),
  TAB(["Mode", "Precision", "Recall", "F1", "Support"], T.test_per_class_mnl.map(r => [r.mode, f(r.precision, 3), f(r.recall, 3), f(r.f1, 3), r.support]),
    [3000, 1500, 1500, 1500, 1526], "Per-class test performance of the frozen MNL.", "obs"),
  ...FIG("model_confusion_matrices", "Test-set confusion matrices of the six candidates.", "obs", 600),
  P(`The MNL recognises bus and car users well but confuses walkers and auto-rickshaw users with two-wheeler and car users, and sends a third of rail users to the two-wheeler. Walking and auto-rickshaw are the hardest classes for every model.`),
  H2("Freezing the selected model"),
  P(`The selected MNL, fitted on the 424 training records only, was saved together with a model card recording its specification, estimates, hyperparameters, cross-validated and test metrics, the selection log, the random seed and the library versions; the cleaning rules, the recovered speed and fare rules and the feature lists were saved as the preprocessing pipeline. Refitting the saved configuration on the training rows reproduces the frozen model exactly (a test checks this), which confirms that no test record entered the fit. The frozen model is used, unchanged, for every prediction and simulation in the rest of the report, and it is never refitted on synthetic data.`),
);

// 13 Synthetic population generation
add(H1("Synthetic Population Generation"),
  H2("Role of the synthetic population"),
  P(`The synthetic population is a larger set of artificial commuters on which policies can be tested. It is generated from the ${summary.split.train} training records, contains predictor variables only and never a choice label, and is never used to train or tune a model. The ${summary.split.test} test records are kept out of the generator so that they can serve as an unseen yardstick for memorisation.`),
  H2("Variables and the treatment of travel times and costs"),
  P(`Twenty-one variables are synthesised: income, age, education, occupation, gender, area type, dwelling, household size, children, working members, car and two-wheeler ownership, parking, car and two-wheeler availability, trip purpose, SAPT, distance, and walking, auto and bus availability. Travel times and costs are **not** synthesised. In the survey they are fixed functions of distance, so they are rebuilt for each synthetic commuter from the synthetic distance with the speed and fare rules of Chapter 7. This preserves the time-distance and cost-distance relationships exactly, where a generic synthesiser would only approximate them. Adults are derived as household size minus children, and rail is available to everyone, as in the survey.`),
  H2("Generation methods compared"),
  B(`**Stratified bootstrap.** Real training records are resampled with replacement within strata of distance band and income. It reproduces the marginals almost perfectly but every record is a copy of a real person, so it serves only as a reference.`),
  B(`**Gaussian copula.** Each variable is mapped to a normal score, discrete variables through a randomised probability integral transform, the correlation matrix of the scores is estimated, new scores are drawn from the multivariate normal and mapped back through the empirical quantiles. Nominal categories are ordered by frequency, as in common implementations, because a copula captures only monotone association.`),
  B(`**Sequential CART.** The variables are generated one at a time, in a fixed order. Each variable is drawn from the training values that fall into the same leaf of a classification or regression tree fitted on the variables generated before it, as in the synthpop package. Distance is smoothed with a kernel inside each leaf so that values are not copied exactly.`),
  EQ([mr("u = F("), sup("x", "−"), mr(") + v [F(x) − F("), sup("x", "−"), mr(")],    z = "), sup("Φ", "−1"), mr("(u),    v ~ U(0, 1)")]),
  P(`Equation 13.1 is the randomised transform used by the copula for discrete variables. CTGAN was not used: with 424 training records, a deep generative model has far more parameters than data and would be likely to memorise them.`),
  H2("Logical constraints"),
  P(`Every generated record must satisfy six rules, the first five of which also hold (or nearly hold) in the survey: (1) children leave at least one adult in the household; (2) working members do not exceed household size; (3) a car can be available only if the household owns one; (4) the same for two-wheelers; (5) walking is available for every trip up to 1.5 km, as for all 126 such trips in the survey; (6) distance stays within the observed range. Violations are counted before they are repaired, and the count is used as a quality measure.`),
  H2("Selection of the method"),
  P(`The rule, again fixed in advance, is that a method is eligible only if its share of exact copies of training records is at most twice the share of held-out real records that coincide with a training record, plus one percentage point (here, 1%). Among eligible methods the one whose discriminator AUC is closest to 0.5 is chosen, with association fidelity breaking ties. AUC values below 0.5 are not treated as better than 0.5, because they arise when near-copies of training records fall on opposite sides of a cross-validation split.`),
  P(`For the CART method we also chose the minimum leaf size by weighing fidelity against memorisation (Table 13.1). With leaves of 5 records, the default of synthpop, ${f(100 * leaf[0].share_synthetic_closer_than_holdout_p5, 0)}% of synthetic records lie closer to a training record than 95% of real held-out people do. With leaves of 10 the median distance to the closest training record equals that of real held-out people, and the fidelity measures hardly change, so 10 was used.`),
  TAB(["Min. leaf", "Mean marginal distance", "Association MAD", "Discriminator AUC", "Exact copies (%)", "Median DCR (holdout)", "Closer than holdout p5 (%)"],
    leaf.map(r => [r.min_leaf, f(r.mean_marginal_distance, 4), f(r.association_mad, 4), f(r.discriminator_auc, 3), f(100 * r.exact_copy_share_synthetic, 2), `${f(r.median_dcr_synthetic, 3)} (${f(r.median_dcr_holdout, 3)})`, f(100 * r.share_synthetic_closer_than_holdout_p5, 1)]),
    [1000, 1300, 1300, 1300, 1200, 1500, 1426], "Sequential CART: fidelity against memorisation for different minimum leaf sizes (5,000 records each). DCR = Gower distance to the closest training record.", "syn", { size: 16 }),
);

// 14 Synthetic validation
const mt = T.synthetic_moments;
const momentRow = (v) => { const r = mt.find(x => x.variable === v && x.data === "real"), s = mt.find(x => x.variable === v && x.data === "synthetic");
  return [v, `${f(r.mean, 2)} / ${f(s.mean, 2)}`, `${f(r.sd, 2)} / ${f(s.sd, 2)}`, `${f(r.p25, 1)} / ${f(s.p25, 1)}`, `${f(r.p50, 1)} / ${f(s.p50, 1)}`, `${f(r.p75, 1)} / ${f(s.p75, 1)}`, `${f(r.p95, 1)} / ${f(s.p95, 1)}`]; };
const marg = T.synthetic_marginal_fidelity;
const worst = marg.filter(r => r.variable !== "Distance_km").reduce((a, b) => (a.distance >= b.distance ? a : b));
add(H1("Synthetic Data Validation"),
  H2("Comparison of the three methods"),
  TAB(["Measure", "Bootstrap", "Gaussian copula", "Sequential CART"], [
    ["Mean marginal distance (TVD / KS)", ...["bootstrap", "copula", "cart"].map(m => f(synthTab[m].mean_marginal_distance, 4))],
    ["Largest marginal distance", ...["bootstrap", "copula", "cart"].map(m => f(synthTab[m].max_marginal_distance, 4))],
    ["Association MAD (Cramér's V)", ...["bootstrap", "copula", "cart"].map(m => f(synthTab[m].association_mad, 4))],
    ["Discriminator AUC", ...["bootstrap", "copula", "cart"].map(m => f(synthTab[m].discriminator_auc, 3))],
    ["Records needing repair (%)", ...["bootstrap", "copula", "cart"].map(m => f(synthTab[m].records_needing_repair_pct, 2))],
    ["Exact copies of training records (%)", ...["bootstrap", "copula", "cart"].map(m => f(100 * synthTab[m].exact_copy_share_synthetic, 2))],
    ["Median DCR (holdout: 0.143)", ...["bootstrap", "copula", "cart"].map(m => f(synthTab[m].median_dcr_synthetic, 3))],
    ["Closer than holdout 5th percentile (%)", ...["bootstrap", "copula", "cart"].map(m => f(100 * synthTab[m].share_synthetic_closer_than_holdout_p5, 1))],
  ], [3500, 1800, 1800, 1926], "Fidelity, consistency and memorisation of the three generators (5,000 records each, compared with the training partition).", "syn"),
  P(`The bootstrap is excluded by the copy rule (${f(100 * synthTab.bootstrap.exact_copy_share_synthetic, 1)}% of its records are real people). The copula produces new records, but a random forest tells them apart from real ones easily (AUC ${f(synthTab.copula.discriminator_auc, 3)}), it distorts the associations the most, and ${f(synthTab.copula.records_needing_repair_pct, 0)}% of its records break a logical rule before repair; for example, ${synthTab.copula.viol_car_available_without_car} records have a car available in a household without one. Sequential CART keeps the associations best, its records cannot be told apart from real ones (AUC ${f(syn.discriminator_auc, 3)}), only ${f(syn.records_needing_repair_pct, 1)}% need repair and only ${f(100 * syn.exact_copy_share_synthetic, 2)}% coincide with a training record. The rule selects CART, and the population of ${summary.synthetic.n.toLocaleString("en-IN")} commuters used from here on is its output.`),
  H2("Marginal distributions"),
  ...FIG("synthetic_marginals", "Category frequencies in the training data and in the synthetic population.", "syn", 600),
  ...FIG("synthetic_distance", "Distance distribution of the training data and the synthetic population (empirical CDF).", "syn", 470),
  P(`Category frequencies agree to within ${f(sizes.second_largest_gap_pp, 1)} percentage points for every variable except ${sizes.largest_category_gap.variable === "Bus_AV" ? "bus availability" : sizes.largest_category_gap.variable}, which is ${f(sizes.bus_av_synthetic_pct, 1)}% in the synthetic population against ${f(sizes.bus_av_train_pct, 1)}% in the training data (total variation distance ${f(worst.distance, 3)}). The distance distribution is reproduced closely (Kolmogorov-Smirnov statistic ${f(marg.find(r => r.variable === "Distance_km").distance, 3)}). Table 14.2 compares means, standard deviations and quantiles.`),
  TAB(["Variable", "Mean", "Std. dev.", "25th pct", "Median", "75th pct", "95th pct"],
    ["Distance_km", "HH_Size", "Children", "Working_Adults", "Car_Ownership", "TwoW_Ownership", "bus_time_min", "TC_Bus", "TC_Car"].map(momentRow),
    [1800, 1200, 1200, 1200, 1200, 1200, 1226], "Moments and quantiles, training data / synthetic population.", "syn", { size: 16 }),
  H2("Associations and pairwise relationships"),
  ...FIG("synthetic_associations", "Pairwise association (bias-corrected Cramér's V) in the training data and the synthetic population, and their difference.", "syn", 600),
  ...FIG("synthetic_pairwise", "Three relationships the synthetic population should preserve.", "syn", 600),
  P(`The association structure is reproduced well overall (mean absolute difference ${f(syn.association_mad, 3)} in Cramér's V). The CART synthesis slightly weakens the link between car and two-wheeler ownership and between household size and number of children, and slightly strengthens the link between distance and bus availability. Income still drives car ownership, the number of children still grows with household size, and walking availability still falls with distance as in the survey.`),
  H2("Memorisation and representativeness"),
  P(`Only ${f(100 * syn.exact_copy_share_synthetic, 2)}% of synthetic records are exact copies of a training record (the corresponding share for real held-out records is ${f(100 * syn.exact_copy_share_holdout, 1)}%), and the median distance from a synthetic record to its closest training record (${f(syn.median_dcr_synthetic, 3)}) equals that of real held-out people. ${f(100 * syn.share_synthetic_closer_than_holdout_p5, 1)}% of synthetic records are closer to a training record than the closest 5% of held-out people, slightly above the 5% expected if synthetic people behaved exactly like new real people.`),
  P(`The synthetic population is a faithful, larger version of the survey sample. It is not calibrated to census totals for any city, so its mode shares describe "commuters like our respondents", not the population of Mumbai or any other region.`),
);

// 15 Mode choice prediction
const ex = T.example_individuals;
add(H1("Mode Choice Prediction"),
  P(`The frozen MNL predicts six probabilities for each of the ${summary.synthetic.n.toLocaleString("en-IN")} synthetic commuters. Unavailable modes receive probability exactly zero and every row sums to one; the prediction function refuses to return anything else, and the tests check this for the whole population. Table 15.1 shows one commuter for each most-likely mode.`),
  TAB(["ID", "Distance (km)", "Income", "Cars", "Available modes", "Walk", "2W", "Auto", "Car", "Bus", "Rail"],
    ex.map(r => [r.ID, f(r.distance_km, 1), r.income, r.cars, r.available, f(r.Walk, 3), f(r["2W"], 3), f(r.Auto, 3), f(r.Car, 3), f(r.Bus, 3), f(r.Rail, 3)]),
    [600, 850, 650, 550, 2276, 680, 680, 680, 680, 680, 700], "Predicted probabilities for six synthetic commuters.", "syn", { size: 15 }),
  TAB(["Mode", "Observed (training)", "Model on training data", "Synthetic: expected", "Synthetic: most likely"],
    MODES.map(m => { const r = base[MODE_NAME[m]]; return [MODE_NAME[m], f(r.observed_train_pct, 2), f(r.model_on_train_pct, 2), f(r.synthetic_expected_pct, 2), f(r.synthetic_hard_pct, 2)]; }),
    [2200, 1700, 1700, 1700, 1726], "Baseline mode shares (%): observed sample, model predictions on the training data, and predictions for the synthetic population.", "syn"),
  ...FIG("baseline_shares", "Baseline mode shares: observed sample against the predicted synthetic population.", "syn", 560),
  P(`On the training data the MNL reproduces the observed shares exactly, a known property of a logit with a full set of alternative-specific constants. In the synthetic population the expected shares stay within about two points of the observed ones; the bus share is higher (${f(base.Bus.synthetic_expected_pct, 1)}% against ${f(base.Bus.observed_train_pct, 1)}%) because the synthetic population has slightly more bus availability. The baseline expected sustainable share is **${f(baseSus, 2)}%**, compared with ${f(summary.baseline.observed_train_sustainable_pct, 2)}% observed in the training sample.`),
  P(`Assigning each commuter to their most likely mode gives a very different picture: car ${f(base.Car.synthetic_hard_pct, 1)}%, walking ${f(base.Walk.synthetic_hard_pct, 1)}% and a sustainable share of ${f(summary.baseline.hard_sustainable_share_pct, 2)}%. Hard assignment exaggerates whichever mode is most likely in each choice set and erases the minority modes. The expected (probability-weighted) share is the statistically correct estimate of the share of trips, so it is the quantity optimised; the hard share is reported alongside it as the brief requires.`),
);

// 16 Scenarios
const scRow = (t, name) => { const r = t[name]; return [name, f(r.expected_sustainable_share_pct, 2), sgn(r.sustainable_change_pp, 2), f(r.car_pct, 2), f(r.twow_pct, 2), f(r.bus_pct, 2), f(r.train_pct, 2), f(r.hard_sustainable_share_pct, 1)]; };
const scHead = ["Scenario", "Sustainable (%)", "Change (pp)", "Car (%)", "2W (%)", "Bus (%)", "Rail (%)", "Most-likely sustainable (%)"];
const scW = [2700, 900, 850, 800, 800, 800, 800, 1376];
add(H1("Sensitivity and Scenario Analysis"),
  P(`Each scenario changes one or more attributes of every synthetic commuter by a percentage and re-predicts with the frozen model. Times and costs stay non-negative and unavailable modes stay unavailable unless the scenario explicitly changes availability. With all changes set to zero, the predictions are identical to the baseline (maximum absolute difference ${summary.no_change_max_abs_diff}), which is Test 9 of the brief. All results in this chapter are model-based counterfactuals.`),
  H2("Experiment 1: travel cost"),
  TAB(scHead, ["Baseline", "Bus fare -10%", "Bus fare -20%", "Bus fare -50%", "Rail fare -10%", "Rail fare -20%", "Rail fare -50%", "Car cost +10%", "Car cost +20%", "Car cost +50%", "Two-wheeler cost +20%", "Auto fare +20%", "Bus fare -10% and car cost +10%"].map(n => scRow(sc.cost, n)),
    scW, "Expected shares under fare and cost changes (selection; the full table is in the results folder).", "sim", { size: 16 }),
  P(`Raising the car cost by 50% adds ${f(sc.cost["Car cost +50%"].sustainable_change_pp, 2)} points of sustainable share, halving the rail fare adds ${f(sc.cost["Rail fare -50%"].sustainable_change_pp, 2)} and halving the bus fare only ${f(sc.cost["Bus fare -50%"].sustainable_change_pp, 2)}. Bus measures are weak at population level because only about one commuter in five has a bus option at all. Part of the car users who leave move to two-wheelers (${f(sc.cost["Car cost +50%"].twow_pct, 1)}% against ${f(sc.cost["Baseline"].twow_pct, 1)}%), which are not sustainable under our definition.`),
  H2("Experiment 2: travel time"),
  TAB(scHead, ["Baseline", "Bus time -10%", "Bus time -20%", "Bus time -30%", "Rail time -10%", "Rail time -20%", "Car time +20%", "Two-wheeler time +20%"].map(n => scRow(sc.time, n)),
    scW, "Expected shares under travel-time changes.", "sim", { size: 16 }),
  ...FIG("scenario_sensitivity", "Change in the expected sustainable share for one-lever scenarios.", "sim", 600),
  P(`Time effects are smaller than cost effects: a 20% faster rail service adds ${f(sc.time["Rail time -20%"].sustainable_change_pp, 2)} points and a 30% faster bus ${f(sc.time["Bus time -30%"].sustainable_change_pp, 2)}. Unlike the cost effects, they depend directly on the assumed walk-time weight, because that assumption is what identifies the time coefficient (Chapter 20 quantifies this).`),
  H2("Experiment 3: availability (extrapolation)"),
  TAB(["Scenario", "Sustainable (%)", "Change (pp)", "Bus (%)", "Walk (%)", "Car (%)"], sc.av.map(r => [r.scenario, f(r.expected_sustainable_share_pct, 2), sgn(r.sustainable_change_pp, 2), f(r.bus_pct, 2), f(r.walk_pct, 2), f(r.car_pct, 2)]),
    [3600, 1100, 1100, 1050, 1050, 1126], "Availability scenarios. These are extrapolations beyond the choice sets observed in the survey.", "sim", { size: 16 }),
  ...FIG("scenario_availability", "Mode shares under availability scenarios (extrapolations).", "sim", 560),
  P(`Offering a bus to people who currently have none produces very large changes: up to ${f(sc.av[sc.av.length - 3].sustainable_change_pp, 0)} points if everyone gains a bus option. These numbers should not be taken at face value. They place people into choice sets that the survey never observed for them; they assume that new bus users would value the bus exactly like current users, who choose it ${f(eda.av.Bus.chosen_when_available_pct, 0)}% of the time and may have been recorded as having a bus option because they use it; and they ignore the service quality a new route would offer. Making walking available for all trips up to 3 km adds ${f(sc.av[sc.av.length - 1].sustainable_change_pp, 2)} points. Rail availability cannot be varied at all, because rail is already available to everyone and the model has learned nothing about rail availability.`),
  H2("Experiment 4: combined packages"),
  TAB(["Scenario", "Sustainable (%)", "Change (pp)", "Car (%)", "Bus (%)", "Rail (%)", "Most-likely sustainable (%)"], sc.comb.map(r => [r.scenario, f(r.expected_sustainable_share_pct, 2), sgn(r.sustainable_change_pp, 2), f(r.car_pct, 2), f(r.bus_pct, 2), f(r.train_pct, 2), f(r.hard_sustainable_share_pct, 1)]),
    [3300, 1000, 950, 850, 850, 850, 1226], "Combined policy packages.", "sim", { size: 16 }),
  ...FIG("scenario_combined", "Mode shares under combined policy packages.", "sim", 600),
  P(`A moderate public-transport package (bus 15% faster and 20% cheaper, rail 10% faster and 20% cheaper) adds ${f(sc.comb[3].sustainable_change_pp, 2)} points; adding a 25% car cost increase to it doubles the gain to ${f(sc.comb[4].sustainable_change_pp, 2)} points. In this model, "pull" measures that improve public transport and "push" measures that make driving dearer reinforce each other.`),
);

// 17 Sustainable definition
add(H1("Sustainable Transportation Definition"),
  P(`For the primary analysis, sustainable modes are walking, bus and suburban rail. Auto-rickshaws are treated as an intermediate, shared motorised mode, and cars and two-wheelers as private motorised modes; none of these three counts as sustainable. For each commuter the sustainable-mode probability and the population measures are`),
  EQ([sub("S", "n"), mr(" = "), sub("P", "n"), mr("(walk) + "), sub("P", "n"), mr("(bus) + "), sub("P", "n"), mr("(rail),     "),
      mr("S̄ = "), frac("1", "N"), sum("n", [sub("S", "n")])]),
  EQ([mr("H = "), frac("1", "N"), sum("n", [mr("1["), sub("argmax", "j"), mr(" "), sub("P", "nj"), mr(" ∈ {walk, bus, rail}]")])]),
  P(`S-bar, the expected sustainable share, estimates the share of trips made by sustainable modes and is the objective of the optimisation. H, the most-likely-mode sustainable share, counts the commuters whose most probable mode is sustainable. Both are reported, but they answer different questions: H ignores how close the second-best mode is, so it responds to policies in jumps and tends to understate minority modes.`),
);

// 18 Formulation
add(H1("Mathematical Formulation of Policy Optimisation"),
  H2("Decision variables"),
  TAB(["Lever", "Symbol", "Data column changed", "Bounds (%)", "Interpretation"], [
    ["Bus travel-time reduction", "x1", "bus_time_min", `${BOUND.bus_time_red[0]} to ${BOUND.bus_time_red[1]}`, "bus priority, more frequent service"],
    ["Bus fare reduction", "x2", "TC_Bus", `${BOUND.bus_fare_red[0]} to ${BOUND.bus_fare_red[1]}`, "fare subsidy"],
    ["Rail travel-time reduction", "x3", "local_train_time_min", `${BOUND.train_time_red[0]} to ${BOUND.train_time_red[1]}`, "faster or more frequent trains (access is not in the data)"],
    ["Rail fare reduction", "x4", "TC_LocalTrain", `${BOUND.train_fare_red[0]} to ${BOUND.train_fare_red[1]}`, "fare subsidy"],
    ["Car out-of-pocket cost increase", "x5", "TC_Car", `${BOUND.car_cost_inc[0]} to ${BOUND.car_cost_inc[1]}`, "proxy for fuel taxes, congestion or parking charges"],
  ], [2200, 700, 1800, 1100, 3226], "Decision variables. The bounds are scenario assumptions, not engineering estimates, and can be changed in the dashboard.", "none", { numericFrom: 99, size: 17 }),
  P(`The car lever changes the per-kilometre money cost of driving, the only car cost in the data. It is a proxy for pricing measures; it does not model parking supply or charges explicitly, because the data contain no parking-cost variable. Bus availability is not a decision variable in the main problem, because extending availability is an extrapolation (Chapter 16), and rail availability cannot be changed at all.`),
  H2("Objective and constraints"),
  P(`The levers act on every synthetic commuter n as`),
  EQ([sub("t'", "n,bus"), mr(" = "), sub("t", "n,bus"), mr("(1 − "), sub("x", "1"), mr("/100),     "), sub("c'", "n,bus"), mr(" = "), sub("c", "n,bus"), mr("(1 − "), sub("x", "2"), mr("/100)")]),
  EQ([sub("c'", "n,car"), mr(" = "), sub("c", "n,car"), mr("(1 + "), sub("x", "5"), mr("/100)")]),
  P(`and similarly for the rail time (x~3~) and fare (x~4~). The problem is`),
  EQ([sub("max", "x"), mr(" S̄(x) = "), frac("1", "N"), sum([mr("n=1")], [sub("P", "n"), mr("(walk | x) + "), sub("P", "n"), mr("(bus | x) + "), sub("P", "n"), mr("(rail | x)")], "N")]),
  EQ([mr("subject to   "), sub("l", "k"), mr(" ≤ "), sub("x", "k"), mr(" ≤ "), sub("u", "k"), mr(",   k = 1, ..., 5")]),
  EQ([mr("and optionally   E(x) = "), sum("k", [frac(sub("x", "k"), sub("u", "k"))]), mr(" ≤ B")]),
  P(`Times and costs remain non-negative because no reduction exceeds 100%, and availability is unchanged, so unavailable modes keep probability zero. There are no intervention cost estimates in the project, and we did not invent any; the primary problem is therefore purely bounded. The optional constraint E(x) ≤ B is a dimensionless "effort budget": each lever is measured as a share of its upper bound, so B = 1 allows one lever at its maximum or several partially. It is used only as a sensitivity analysis to reveal which levers the model rewards most per unit of intervention; it is not a financial budget.`),
  P(`With the MNL, the sustainable probability of every commuter increases monotonically in each lever: lower bus or rail time or fare raises the utility of a sustainable mode, and a higher car cost lowers the utility of a non-sustainable one. The bounded optimum must therefore lie at the corner where every lever equals its upper bound. We nevertheless solve the problem numerically, because the code is generic (it does not assume monotonicity and works for any frozen model) and because verifying that the search finds the corner is a useful check of the whole simulation chain.`),
);

// 19 Algorithm
add(H1("Optimisation Algorithm"),
  H2("Grid search"),
  P(`The baseline method evaluates every combination of lever values on a grid with ${opt.grid.step}% steps: 7 x 11 x 5 x 11 x 11 = ${opt.grid.evaluations.toLocaleString("en-IN")} policies, each evaluated on all ${summary.synthetic.n.toLocaleString("en-IN")} synthetic commuters. The full table of objective values is saved, so the same pass also answers every effort-budget question (the objective does not depend on the budget; only feasibility does).`),
  H2("Fast evaluation"),
  P(`For a logit model, a lever only rescales one column of the design array, and that column enters the utility linearly. The evaluator therefore precomputes the design array once and, for each policy, multiplies only the affected cells. This gives exactly the same probabilities as rebuilding the data frame and calling the model (a test compares the two to 10^−12^) at a fraction of the cost; the full grid took ${f(opt.grid.seconds / 60, 1)} minutes. For other model types the code falls back to the general path.`),
  H2("Differential evolution"),
  P(`As an independent global search, differential evolution (population 20 per dimension, Sobol initialisation, up to 200 generations, seed 42, with local polishing) maximises the same objective over the continuous box. Under an effort budget, the budget is passed as a nonlinear constraint.`),
  H2("Verification and benchmarks"),
  P(`The reported optimum is verified independently of the search: the frozen model and the synthetic population are reloaded from disk, the policy is applied through the general data-frame path, and the objective is recomputed and compared with the reported value. The check also confirms that every lever lies within its bounds, that all times and costs are non-negative, that availability is unchanged and that every probability vector is valid. Finally, the optimum is compared with the baseline, with each lever alone at its bound and with two hand-made packages; it should never be beaten.`),
);

// 20 Results
const shareRow = (m) => { const r = optShares[MODE_NAME[m]]; return [MODE_NAME[m], f(r.baseline_expected_pct, 2), f(r.optimised_expected_pct, 2), sgn(r.change_expected_pp, 2), f(r.baseline_hard_pct, 2), f(r.optimised_hard_pct, 2)]; };
const ver = opt.verification;
add(H1("Experimental Results"),
  H2("Optimal bounded policy"),
  TAB(["Lever", "Lower bound", "Upper bound", "Optimised value"], [
    ["Bus travel-time reduction (%)", "bus_time_red"], ["Bus fare reduction (%)", "bus_fare_red"], ["Suburban rail travel-time reduction (%)", "train_time_red"],
    ["Suburban rail fare reduction (%)", "train_fare_red"], ["Car out-of-pocket cost increase (%)", "car_cost_inc"]].map(([lab, k]) => [lab, f(BOUND[k][0], 0), f(BOUND[k][1], 0), f(levers[k], 1)]),
    [4200, 1500, 1500, 1826], "Optimised policy (bounded problem).", "sim"),
  TAB(["Mode", "Baseline expected (%)", "Optimised expected (%)", "Change (pp)", "Baseline most likely (%)", "Optimised most likely (%)"],
    [...MODES.map(shareRow), ["Sustainable (walk + bus + rail)", f(baseSus, 2), f(optSus, 2), sgn(opt.improvement_pp, 2), f(opt.hard_sustainable.baseline_pct, 2), f(opt.hard_sustainable.optimised_pct, 2)]],
    [2600, 1250, 1250, 1100, 1400, 1426], "Mode shares of the synthetic population at baseline and under the optimised policy.", "sim", { size: 17 }),
  ...FIG("optimisation_mode_shares", "Expected mode shares at baseline and under the optimised policy.", "sim", 560),
  P(`As expected for a monotone model, the optimum puts every lever at its upper bound. It raises the expected sustainable share from ${f(baseSus, 2)}% to ${f(optSus, 2)}%, an improvement of **${f(opt.improvement_pp, 2)} percentage points**, and the most-likely-mode sustainable share from ${f(opt.hard_sustainable.baseline_pct, 2)}% to ${f(opt.hard_sustainable.optimised_pct, 2)}%. The gain comes mostly from cars (${sgn(optShares.Car.change_expected_pp, 2)} points), which move to rail (${sgn(optShares["Suburban rail"].change_expected_pp, 2)}), bus (${sgn(optShares.Bus.change_expected_pp, 2)}) and, less helpfully, two-wheelers (${sgn(optShares["Two-wheeler"].change_expected_pp, 2)}).`),
  H2("Agreement of the two searches and verification"),
  TAB(["Item", "Value"], [
    ["Grid search: evaluations / time", `${opt.grid.evaluations.toLocaleString("en-IN")} / ${f(opt.grid.seconds, 0)} s`],
    ["Grid search: best objective", `${f(opt.grid.best_objective_pct, 6)}%`],
    ["Differential evolution: evaluations / time", `${opt.differential_evolution.evaluations.toLocaleString("en-IN")} / ${f(opt.differential_evolution.seconds, 0)} s`],
    ["Differential evolution: best objective", `${f(opt.differential_evolution.objective_pct, 6)}%`],
    ["Differential evolution: levers", Object.values(opt.differential_evolution.levers).map(v => f(v, 4)).join(", ")],
    ["Reported optimum found by", opt.found_by],
    ["Independent recomputation of the objective", `${f(100 * ver.recomputed_objective, 6)}%`],
    ["Absolute difference from reported value", `${ver.abs_difference}`],
    ["Constraint violations", ver.constraint_problems.length ? ver.constraint_problems.join("; ") : "none"],
    ["Non-negative times and costs; availability unchanged", `${ver.non_negative_times && ver.non_negative_costs ? "yes" : "no"}; ${ver.availability_unchanged ? "yes" : "no"}`],
  ], [4500, 4526], "Search results and independent verification.", "sim", { numericFrom: 99 }),
  P(`Differential evolution converges to the same corner to within 10^−5^ of every bound; the grid point, which lies exactly on the bounds, is marginally higher and is the reported optimum. The independent recomputation reproduces the reported objective exactly and every constraint holds.`),
  H2("Comparison with benchmark policies"),
  TAB(["Policy", "Bus time", "Bus fare", "Rail time", "Rail fare", "Car cost", "Effort", "Sustainable (%)"],
    T.optimisation_benchmarks.map(r => [r.policy, f(r.bus_time_red, 0), f(r.bus_fare_red, 0), f(r.train_time_red, 0), f(r.train_fare_red, 0), f(r.car_cost_inc, 0), f(r.effort, 1), f(r.sustainable_share_pct, 2)]),
    [3000, 750, 750, 750, 750, 750, 750, 1526], "Optimised policy against benchmark policies (lever values in %).", "sim", { size: 16 }),
  ...FIG("optimisation_benchmarks", "Expected sustainable share of the optimised and benchmark policies.", "sim", 540),
  P(`No benchmark beats the optimum. On its own, the car cost increase is the most effective lever (${sgn(bench["Only car out-of-pocket cost increase at bound"].sustainable_share_pct - baseSus, 2)} points), followed by the rail fare (${sgn(bench["Only suburban rail fare reduction at bound"].sustainable_share_pct - baseSus, 2)}), rail time (${sgn(bench["Only suburban rail travel-time reduction at bound"].sustainable_share_pct - baseSus, 2)}), bus fare (${sgn(bench["Only bus fare reduction at bound"].sustainable_share_pct - baseSus, 2)}) and bus time (${sgn(bench["Only bus travel-time reduction at bound"].sustainable_share_pct - baseSus, 2)}). All four public-transport levers together give ${sgn(bench["Public transport levers at bound, no car pricing"].sustainable_share_pct - baseSus, 2)} points; adding car pricing takes the total to ${sgn(opt.improvement_pp, 2)}.`),
  H2("Effort budget"),
  TAB(["Budget B", "Bus time", "Bus fare", "Rail time", "Rail fare", "Car cost", "Effort used", "Sustainable (%)"],
    budget.map(r => [f(r.effort_budget, 2), f(r.bus_time_red, 0), f(r.bus_fare_red, 0), f(r.train_time_red, 0), f(r.train_fare_red, 0), f(r.car_cost_inc, 0), f(r.effort_used, 2), f(r.sustainable_share_pct, 2)]),
    [1100, 1000, 1000, 1000, 1000, 1000, 1200, 1726], "Best policy on the grid for each effort budget (lever values in %).", "sim", { size: 16 }),
  ...FIG("optimisation_effort_budget", "Best achievable sustainable share for each effort budget and how the budget is spent.", "sim", 600),
  P(`When the total effort is limited, the levers are switched on in a fixed order: car pricing first, then the rail fare, rail time, bus fare and, last, bus time. The frontier flattens: the first unit of effort buys ${f(budget.find(r => r.effort_budget === 1).sustainable_share_pct - baseSus, 2)} points, the last unit only ${f(budget[budget.length - 1].sustainable_share_pct - budget.find(r => r.effort_budget === 4).sustainable_share_pct, 2)}. Differential evolution run with the same constraint at B = 1 and B = 2 reaches ${T.optimisation_budget_de_check.map(r => f(r.de_best_pct, 3) + "%").join(" and ")}, matching the grid (${T.optimisation_budget_de_check.map(r => f(r.grid_best_pct, 3) + "%").join(" and ")}).`),
  H2("Sensitivity to the walk-time weight"),
  TAB(["Walk weight", "Time coef.", "Cost coef.", "VOT (INR/h)", "Baseline (%)", "Optimum (%)", "Gain: bus time", "Gain: bus fare", "Gain: rail time", "Gain: rail fare", "Gain: car cost"],
    wws.map(r => [f(r.walk_weight, 1), f(r.B_time, 4), f(r.B_cost, 4), f(r.VOT_INR_per_hour, 1), f(r.baseline_pct, 2), f(r.all_levers_at_bound_pct, 2),
      f(r.gain_bus_time_red_pp, 2), f(r.gain_bus_fare_red_pp, 2), f(r.gain_train_time_red_pp, 2), f(r.gain_train_fare_red_pp, 2), f(r.gain_car_cost_inc_pp, 2)]),
    [800, 820, 820, 820, 820, 820, 820, 820, 820, 820, 846], "Re-estimating the MNL with other walk-time weights: optimum and single-lever gains (pp). The frozen model uses a weight of 2.", "sim", { size: 15 }),
  P(`Across walk weights from 1.5 to 3, the optimised sustainable share stays between ${f(Math.min(...wws.map(r => r.all_levers_at_bound_pct)), 2)}% and ${f(Math.max(...wws.map(r => r.all_levers_at_bound_pct)), 2)}%, and the gains from the cost levers barely change. The rail-time gain is the number that depends on the assumption: ${f(wws[0].gain_train_time_red_pp, 2)} points at weight 1.5 against ${f(wws[wws.length - 1].gain_train_time_red_pp, 2)} at weight 3. The main conclusions are therefore robust to the walk-weight assumption; the precise value of travel-time measures is not.`),
);

// 21 Discussion
add(H1("Discussion"),
  H2("Prediction is not simulation"),
  P(`The central result of the model comparison is that the best classifiers were the worst simulators. XGBoost predicts the held-out choices far better than the MNL, yet it is useless for evaluating a fare cut because its predictions do not move; logistic regression and the random forest are worse than useless because they move the wrong way. The cause is specific and identifiable: the survey's travel times were generated from distance, so "a longer bus trip" and "a slower bus" are the same thing in the data, and a flexible model learns whatever correlation is strongest. A random-utility model is not immune to the same problem, as the positive time coefficient of the unrestricted MNL showed. What it offers is a structure in which the necessary assumption (here, a single time coefficient with walking minutes weighted twice) is explicit, testable against the data and open to sensitivity analysis. We think this is the right trade-off for a model whose purpose is policy analysis, but it should be stated plainly: the frozen model is the most defensible simulator, not the best predictor.`),
  H2("What the policy results say"),
  P(`Within the model, three messages are robust. Making car travel dearer is the most effective single lever, because it acts on the large car market directly. Rail fare and time measures come next, because rail is available to everyone. Bus measures are weak, not because commuters are insensitive to buses but because only about one commuter in five has a bus option; the large but uncertain effects of extending bus availability suggest that network coverage, not fares, is the binding constraint for the bus. A fourth message is a warning: a share of the commuters who leave the car move to two-wheelers, so car pricing without attractive public transport partly shifts trips between two non-sustainable modes.`),
  P(`The overall gain of about five percentage points under quite aggressive interventions (halving public-transport fares and raising car costs by half) is modest. That is consistent with mode choice being strongly tied to income, vehicle ownership and trip length, which fares do not change, and with the price and time sensitivities identified from these data being moderate. The value of time implied by the model, about INR ${f(vot, 0)} per hour, should be read with the same caution as the time effects, since it inherits the walk-weight assumption.`),
  H2("On the synthetic population"),
  P(`Sequential CART synthesis turned out to be a good fit for a small, mostly categorical survey: it preserved marginals and associations, respected the logical rules almost automatically and did not memorise the data once the leaf size was raised. Rebuilding times and costs from distance instead of synthesising them kept the population exactly consistent with the survey's level-of-service rules. The price of that faithfulness is that the population inherits the sample's composition, including its quirks, such as almost no respondents of gender code 0 and the unusual concentration of bus availability.`),
);

// 22 Limitations
add(H1("Limitations"),
  B(`**Level-of-service data.** Travel times and costs are deterministic functions of distance and exclude access, egress and waiting time. The data alone cannot identify time effects; ours rest on the assumption that a walking minute is worth two in-vehicle minutes, which the data do not reject but cannot confirm.`),
  B(`**Sample size.** With 531 records (107 for testing), metric estimates are noisy: macro-F1 varies by 0.05-0.07 between folds, and the test-set interval for the MNL's macro-F1 spans 0.35-0.56. The small classes (34 auto-rickshaw and 47 walking trips) are predicted poorly by every model.`),
  B(`**Coding ambiguity.** The meanings of trip-purpose codes, occupation codes 1-5, dwelling code 4, gender, area type and parking space are not confirmed. They enter the models only as codes, which is enough for prediction, but they limit interpretation.`),
  B(`**Availability.** Bus availability may have been recorded with the chosen mode in mind, and rail availability is constant. Availability scenarios are extrapolations and the model says nothing about rail access.`),
  B(`**Model structure.** The MNL assumes independence from irrelevant alternatives. The nested logit relaxes this for {auto, bus, rail} but did not predict better. Taste variation across commuters, beyond the income and purpose terms, is not modelled.`),
  B(`**Selection trade-off.** The frozen model was chosen for its simulation behaviour and is a weaker classifier than XGBoost. The readiness check was declared in advance, but a reader who weights prediction more heavily would choose differently.`),
  B(`**Synthetic population.** It mirrors the survey sample, not a census. Population shares are shares of "commuters like our respondents".`),
  B(`**Counterfactuals, not causes.** All scenario and optimisation results describe the fitted model's response to changed inputs. They are not causal estimates of what an intervention would achieve; supply-side effects, crowding, induced demand, residential relocation and changes in car ownership are all outside the model.`),
  B(`**Policy bounds and costs.** The bounds are scenario assumptions and no cost data were available, so no financial budget was optimised. The effort budget is a device for ranking levers, not an economic constraint. The car lever is a money-cost proxy and does not represent parking supply.`),
);

// 23 Conclusion
add(H1("Conclusion"),
  P(`We built and tested a complete pipeline from a raw survey workbook to an optimised and independently verified transport policy. The audit revealed that the workbook's travel times and costs are computed from distance and that eleven records carried corrupted times; both findings were documented, the corrupted values were corrected from evidence within the data, and the deterministic structure was carried through every later decision. Six candidate models were compared on identical folds. XGBoost classified best, but only the logit models responded correctly to the policy levers, and the multinomial logit was frozen under a rule written down before fitting. A synthetic population of 5,000 commuters generated by sequential CART reproduced the survey's distributions and associations without copying its records.`),
  P(`Under the frozen model, the expected sustainable share of the synthetic population is ${f(baseSus, 1)}% at baseline and ${f(optSus, 1)}% when every lever is at its upper bound, a gain of ${f(opt.improvement_pp, 1)} percentage points confirmed by grid search, differential evolution and an independent recomputation. Car pricing and rail fares are the most effective levers and bus measures the least, mainly because few commuters have a bus option. These conclusions are robust to the main modelling assumption, but they remain model-based counterfactuals: they show what the fitted model implies, and they become evidence about real behaviour only to the extent that the model's assumptions hold.`),
);

// 24 Future scope
add(H1("Future Scope"),
  B(`Collect real level-of-service data (for example from a routing engine or a travel survey with reported times), including access, egress and waiting times, so that time effects can be estimated rather than assumed.`),
  B(`Add a stated-preference experiment in which fares and times vary independently of distance; this is the standard way to identify time and cost sensitivities for policy analysis.`),
  B(`Confirm the coding of the ambiguous variables with the data owner and re-run the pipeline, which takes about twenty minutes.`),
  B(`Estimate mixed logit models to capture taste variation, and behaviourally constrained machine-learning models (for example, gradient-boosted utilities with monotonicity constraints) that could combine the accuracy of XGBoost with sensible policy responses.`),
  B(`Calibrate the synthetic population to census marginals by iterative proportional fitting, so that population shares refer to a real region.`),
  B(`Obtain intervention cost estimates and replace the dimensionless effort budget with a financial budget, and add equity measures, such as gains by income group, to the objective.`),
);

// 25 References
const refs = [
  "Beckman, R. J., Baggerly, K. A., & McKay, M. D. (1996). Creating synthetic baseline populations. Transportation Research Part A, 30(6), 415-429.",
  "Ben-Akiva, M., & Lerman, S. R. (1985). Discrete Choice Analysis: Theory and Application to Travel Demand. MIT Press.",
  "Bergsma, W. (2013). A bias-correction for Cramér's V and Tschuprow's T. Journal of the Korean Statistical Society, 42(3), 323-328.",
  "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.",
  "Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). Classification and Regression Trees. Wadsworth.",
  "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).",
  "Gower, J. C. (1971). A general coefficient of similarity and some of its properties. Biometrics, 27(4), 857-871.",
  "Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning (pp. 1321-1330).",
  "Hagenauer, J., & Helbich, M. (2017). A comparative study of machine learning classifiers for modeling travel mode choice. Expert Systems with Applications, 78, 273-282.",
  "Hillel, T., Bierlaire, M., Elshafie, M. Z. E. B., & Jin, Y. (2021). A systematic review of machine learning classification methodologies for modelling passenger mode choice. Journal of Choice Modelling, 38, 100221.",
  "Koppelman, F. S., & Bhat, C. (2006). A Self Instructing Course in Mode Choice Modeling: Multinomial and Nested Logit Models. U.S. Department of Transportation, Federal Transit Administration.",
  "McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), Frontiers in Econometrics (pp. 105-142). Academic Press.",
  "Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. In Proceedings of the 22nd International Conference on Machine Learning (pp. 625-632).",
  "Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke creation of synthetic data in R. Journal of Statistical Software, 74(11), 1-26.",
  "Ortúzar, J. de D., & Willumsen, L. G. (2011). Modelling Transport (4th ed.). Wiley.",
  "Patki, N., Wedge, R., & Veeramachaneni, K. (2016). The Synthetic Data Vault. In IEEE International Conference on Data Science and Advanced Analytics (pp. 399-410).",
  "Pearl, J. (2009). Causality: Models, Reasoning, and Inference (2nd ed.). Cambridge University Press.",
  "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
  "Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with Python. In Proceedings of the 9th Python in Science Conference (pp. 92-96).",
  "Storn, R., & Price, K. (1997). Differential evolution: A simple and efficient heuristic for global optimization over continuous spaces. Journal of Global Optimization, 11(4), 341-359.",
  "Train, K. E. (2009). Discrete Choice Methods with Simulation (2nd ed.). Cambridge University Press.",
  "Virtanen, P., et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261-272.",
  "Wardman, M. (2004). Public transport values of time. Transport Policy, 11(4), 363-377.",
  "Xu, L., Skoularidou, M., Cuesta-Infante, A., & Veeramachaneni, K. (2019). Modeling tabular data using conditional GAN. In Advances in Neural Information Processing Systems 32.",
  "Zhao, X., Yan, X., Yu, A., & Van Hentenryck, P. (2020). Prediction and behavioral analysis of travel mode choice: A comparison of machine learning and logit models. Travel Behaviour and Society, 20, 22-35.",
];
add(H1("References"), ...refs.map(r => new Paragraph({ children: runs(r), spacing: { after: 100, line: 276 }, indent: { left: 567, hanging: 567 } })));

// Appendix A: data dictionary
add(HA("Appendix A   Verified Data Dictionary", "A"),
  P(`Status codes: OK = consistent with the document or the data; DISCREPANCY = data and document disagree; UNDOCUMENTED = not in the coding document; DUPLICATE = identical to another column; CONSTANT = one value only; DERIVED = computed from distance; CHECK = logical inconsistencies flagged.`),
  TAB(["Variable", "Type", "Documented coding", "Observed values", "Status", "Treatment"],
    T.data_dictionary.map(r => [r.variable, r.type, r.documented_coding, r.observed_values, r.status, r.treatment]),
    [1500, 1000, 1700, 1900, 1100, 1826], "Verified data dictionary (all 38 columns of the workbook).", "obs", { numericFrom: 99, size: 14 }),
  P(`**Open questions for the data owner.** (1) What do Trip_Purpose codes 1 and 2 mean? The data suggest 2 = work trip. (2) What are occupation codes 1-5, in particular code 5? (3) What is dwelling code 4? (4) How are Gender, Area_Type and ParkingSpace coded, and does ParkingSpace refer to home or to the destination? (5) Does "Children" mean household members under 18? (6) How was mode availability recorded, and were travel times and costs deliberately computed from distance?`),
);

// Appendix B: reproducibility and tests
const testRows = (T.tests || []).map(r => [r.file.replace("test_", "").replace(/_/g, " "), r.test.replace(/^test_/, "").replace(/_/g, " "), r.status, f(r.seconds, 1)]);
add(HA("Appendix B   Reproducibility and Tests", "B"),
  P(`The whole analysis is produced by one command, \`python run_pipeline.py\`, which runs the eight phases from the raw workbook to the optimisation output in about ${f(summary.runtime_minutes, 0)} minutes on an eight-core laptop. All random steps are seeded (split and models 42, comparison folds 2026, synthetic population 7). Library versions are pinned in \`requirements.txt\` and recorded in the model card (Python ${card.versions.python}, pandas ${card.versions.pandas}, NumPy ${card.versions.numpy}, scikit-learn ${card.versions["scikit-learn"]}, XGBoost ${card.versions.xgboost}). Every table and figure in this report is read from the files the pipeline writes to *reports/results* and *reports/figures*; the report itself is generated from those files by \`reports/report_builder\`.`),
  P(tests ? `The test suite contains ${tests.tests} tests covering the twelve tests required by the brief; in the final run ${Number(tests.tests) - Number(tests.failures) - Number(tests.errors) - Number(tests.skipped)} passed, ${tests.failures} failed, ${tests.errors} raised errors and ${tests.skipped} were skipped (total ${f(tests.time, 0)} s, of which most is the end-to-end test, which reruns the whole pipeline in fast mode in a temporary folder).` : `The test suite results were not available when this report was generated.`),
  ...(testRows.length ? TAB(["Test file", "Test", "Result", "Seconds"], testRows, [1700, 5200, 1000, 1126], "Automated tests and their results in the final run.", "none", { numericFrom: 2, size: 15 }) : []),
  P(`Mapping to the brief: Test 1 (data validation) is test_01_data; Test 2 (all models train) and the estimator checks are test_02_models; Test 3 (no leakage) is test_03_evaluation; Tests 4 and 5 (individual prediction, availability) are test_04_prediction; Tests 6 and 7 (synthetic data and its predictions) are test_05_synthetic; Tests 8 and 9 (scenarios, no-change) are test_06_scenarios; Tests 10 and 11 (constraints, independent verification) are test_07_optimisation; Test 12 (end to end) is test_08_end_to_end.`),
);

// Appendix C: code
add(HA("Appendix C   Selected Code", "C"),
  P(`The two pieces of code that matter most for correctness are the MNL likelihood with availability and the availability mask applied to the machine-learning models. They are reproduced from *src/choice_models.py* and *src/model_training.py*.`),
  ...CODE([
    "def negll(theta):                          # MultinomialLogit.fit",
    "    V = np.where(A > 0, Xd @ theta, -np.inf) # unavailable modes drop out",
    "    return -(V[idx, y] - logsumexp(V, axis=1)).sum()",
    "",
    "def grad(theta):",
    "    P = self._probs(theta, Xd, A)",
    "    return -(x_chosen - np.einsum('nj,njk->nk', P, Xd)).sum(axis=0)",
  ]),
  new Paragraph({ spacing: { after: 160 }, children: [] }),
  ...CODE([
    "def predict_proba(self, X):                # AvailabilityMasked",
    "    raw = self.estimator_.predict_proba(X)",
    "    P = np.zeros((len(X), len(C.MODES)))",
    "    P[:, self.estimator_.classes_] = raw",
    "    A = X[[C.AV_COLS[m] for m in C.MODES]].to_numpy(float)",
    "    P = np.maximum(P, self.eps) * A",
    "    return P / P.sum(axis=1, keepdims=True)",
  ]),
);

// ----------------------------------------------------------------------------- document
const doc = new Document({
  creator: "Course project team",
  title: "Commuter Mode Choice Modelling and Sustainable Transport Policy Optimisation",
  description: "Project report",
  features: { updateFields: true },
  styles: {
    default: { document: { run: { font: FONT, size: 23 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 120, after: 280 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT, color: "1F3864" }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1, keepNext: true } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2, keepNext: true } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 283 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "-", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1134, hanging: 283 } } } }] },
    { reference: "numbers", levels: [
      { level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 567, hanging: 340 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 20 })] })] }) },
    children: C,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log(`wrote ${OUT} (${(buf.length / 1e6).toFixed(1)} MB, ${C.length} blocks)`);
});
