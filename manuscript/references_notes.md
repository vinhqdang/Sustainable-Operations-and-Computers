# Reference notes

Verification: every DOI entry in `references.bib` was checked against Crossref metadata (title, authors, year, venue, volume/issue/pages, DOI) on 29 September 2026. NeurIPS/JMLR entries without DOIs were checked against the publisher pages (proceedings.neurips.cc, jmlr.org) and, where available, the arXiv export API. Books without Crossref records were checked against Open Library ISBN records. Summaries are condensed from the abstracts (Crossref, OpenAlex or Semantic Scholar).

## Data-centre energy use and emissions of computing

- `masanet2020recalibrating` -- Science 367(6481). Bottom-up re-estimate of global data-centre energy use showing that, despite large growth in computing output between 2010 and 2018, energy use rose only modestly owing to efficiency gains; argues that policy is needed to sustain this in the near term.
- `freitag2021real` -- Patterns 2(9). Critiques estimates of ICT's climate impact; puts ICT's share of global GHG emissions at 1.8-2.8%, possibly 2.1-3.9% after correcting supply-chain truncation, and argues emissions will rise without concerted intervention.
- `shehabi2016united` -- LBNL/OSTI report. Estimates US data-centre electricity use 2000-2020: about 70 billion kWh in 2014 (about 1.8% of US electricity), growing only about 4% over 2010-2014 thanks to efficiency.
- `aslan2018electricity` -- J. Industrial Ecology 22(4). Reviews estimates of the electricity intensity of fixed-line internet data transmission (kWh/GB), shows that system boundary, assumptions and data year drive the spread, and derives criteria for accurate estimates.
- `strubell2019energy` -- ACL 2019. Quantifies the financial and carbon cost of training large NLP models and recommends reporting training cost and prioritising efficient methods.
- `patterson2021carbon` -- arXiv:2104.10350 (preprint). Estimates energy and CO2e of training several large neural networks and shows that the choice of datacentre location, hardware and model architecture can change the footprint by large factors.
- `patterson2022carbon` -- IEEE Computer 55(7). Presents four best practices (model, machine, mechanisation, map/location) that reduce ML training energy and emissions, predicting that training emissions will plateau and then fall by 2030.
- `lacoste2019quantifying` -- arXiv:1910.09700 (preprint). Discusses factors driving ML training emissions (grid location, hardware, training time) and introduces an emissions calculator for practitioners.
- `dodge2022measuring` -- ACM FAccT 2022. Measures the operational carbon intensity of machine-learning workloads in cloud instances using location- and time-specific marginal emissions data, and evaluates shifting jobs across regions and times (e.g., pausing and resuming) as a way to cut emissions.

## Carbon-aware computing and temporal/spatial load shifting

- `wiesner2021lets` -- Middleware 2021. Analyses how much delay-tolerant cloud workloads can reduce emissions by temporal shifting in several regions (including Great Britain), using carbon-intensity data and forecasts; finds that savings depend strongly on the region and on the allowed delay.
- `radovanovic2023carbon` -- IEEE Trans. Power Systems 38(2). Describes Google's Carbon-Intelligent Compute Management, which uses next-day carbon-intensity forecasts, day-ahead demand predictions and risk-aware optimisation to set hourly Virtual Capacity Curves that delay temporally flexible workloads to greener hours while preserving daily capacity.
- `zheng2020mitigating` -- Joule 4(10). Simulates migrating data-centre load from fossil-heavy PJM to CAISO to absorb curtailed renewables, showing substantial reductions in curtailment and GHG emissions.
- `hanafy2023carbonscaler` -- Proc. ACM Meas. Anal. Comput. Syst. (POMACS/SIGMETRICS) 7(3). Proposes "carbon scaling": elastic batch jobs vary their server allocation with grid carbon intensity to reduce emissions without the long delays of suspend-resume.
- `souza2023ecovisor` -- ASPLOS 2023. Designs an "ecovisor" that virtualises the energy system (grid, solar, batteries) and exposes software-defined control so applications can optimise carbon efficiency.
- `sukprasert2024limitations` -- EuroSys 2024. Data-driven study (carbon-intensity data from 123 regions; batch and interactive workloads) of upper bounds on carbon-aware spatio-temporal workload shifting; finds practical reductions are limited and far from ideal, simple policies yield most of the gains, and the relative benefit shrinks as grids become greener.
- `acun2023carbon` -- ASPLOS 2023. Carbon Explorer framework for designing 24/7 carbon-free datacentres by trading off renewable investment, storage and carbon-aware workload scheduling, including operational versus embodied carbon.
- `bashir2021enabling` -- ACM SoCC 2021. Argues that sustainable clouds need applications to adapt to when and where low-carbon energy is available, and proposes virtualising the energy system to expose energy and carbon information.
- `liu2011greening` -- ACM SIGMETRICS 2011. Shows that geographical load balancing driven by electricity price can increase total energy use, and that it can instead be designed to encourage use of green energy and reduce brown energy use; provides distributed algorithms.
- `lin2013dynamic` -- IEEE/ACM Trans. Networking 21(5). Studies dynamic right-sizing (turning off servers at low load) with switching costs; derives the structure of the offline optimum and a 3-competitive "lazy" online algorithm, validated on real traces.
- `rao2010minimizing` -- IEEE INFOCOM 2010. Formulates minimisation of total electricity cost for distributed internet data centres in a multi-electricity-market environment subject to quality-of-service constraints, exploiting time and location price diversity.
- `goiri2011greenslot` -- SC 2011. GreenSlot, a batch job scheduler for a solar-powered datacentre that predicts solar availability and schedules jobs to maximise green energy use while meeting deadlines (up to 117% more green energy, up to 39% lower cost).
- `goiri2012greenhadoop` -- EuroSys 2012. GreenHadoop, a MapReduce framework that delays deferrable background jobs within bounds to match predicted green-energy supply and minimise brown energy cost.
- `wierman2014opportunities` -- IGCC 2014. Survey of data-centre demand response: its potential for integrating renewables and peak shaving, progress to date and open challenges.

## Grid carbon-intensity forecasting and data

- `maji2022carboncast` -- ACM BuildSys 2022. CarbonCast, a hierarchical machine-learning approach giving multi-day (up to 96 h) forecasts of grid carbon intensity from forecasts of the generation mix.
- `bokde2021short` -- Applied Energy 281. Proposes time-series decomposition methods for 48-hour-ahead forecasting of electricity CO2 emissions intensity to support scheduling flexible consumption, benchmarked against state-of-the-art models.
- `leerbeck2020short` -- Applied Energy 277. Machine-learning forecasts of average and marginal CO2 emission intensity for the Danish DK2 zone using hundreds of explanatory variables reduced by LASSO and forward selection.
- `lowry2018day` -- Building Services Eng. Res. Technol. 39(6). Day-ahead forecasting of GB grid carbon intensity to support carbon-targeted demand-response scheduling of HVAC plant.
- `staffell2017measuring` -- Energy Policy 102. Analyses the 46% fall in British electricity-sector emissions (to 2016), attributing it to falling demand, coal-to-gas switching under the carbon price floor, and renewables growth.
- `nesoapi` -- Carbon Intensity API (dataset). NESO, with Environmental Defense Fund Europe, the University of Oxford Department of Computer Science and WWF, provides national and 14-region GB carbon-intensity forecasts (96+ hours ahead) and generation mix; accessed 29 September 2026.

## Conformal prediction and uncertainty quantification

- `vovk2005algorithmic` -- Springer 2005 book. Foundational monograph introducing conformal prediction, which gives prediction sets with guaranteed validity under exchangeability.
- `lei2018distribution` -- JASA 113(523). General framework for distribution-free predictive inference in regression using conformal inference; analyses full and split conformal methods and a jackknife variant with finite-sample marginal coverage.
- `romano2019conformalized` -- NeurIPS 2019 (arXiv:1905.03222). Conformalized quantile regression (CQR): combines quantile regression with split conformal calibration to obtain adaptive, heteroscedasticity-aware intervals with finite-sample coverage.
- `gibbs2021adaptive` -- NeurIPS 2021 (arXiv:2106.00170). Adaptive conformal inference (ACI): an online update of the miscoverage level that achieves long-run target coverage under arbitrary distribution shift.
- `stankeviciute2021conformal` -- NeurIPS 2021. Conformal forecasting RNN (CF-RNN): applies conformal prediction to multi-horizon RNN forecasts to obtain intervals with finite-sample coverage guarantees across horizons.
- `xu2023conformal` -- IEEE TPAMI 45(10). EnbPI: ensemble-based, distribution-free prediction intervals for time series with bounds on conditional and marginal coverage gaps, without requiring exchangeability or data splitting.
- `angelopoulos2023conformal` -- Foundations and Trends in ML 16(4). Tutorial introduction to conformal prediction and its extensions (conformalized quantile regression, distribution shift, risk control) for practitioners.

## Forecasting methods and software

- `friedman2001greedy` -- Annals of Statistics 29(5). Introduces gradient boosting as steepest-descent in function space, with algorithms for least-squares, LAD, Huber and logistic losses and regression-tree base learners.
- `ke2017lightgbm` -- NeurIPS 2017. LightGBM: histogram-based gradient-boosted trees with gradient-based one-side sampling and exclusive feature bundling for fast training (the design followed by scikit-learn's HistGradientBoosting estimators).
- `pedregosa2011scikit` -- JMLR 12. Describes scikit-learn, a Python library of machine-learning algorithms with a consistent API.
- `hyndman2006another` -- Int. J. Forecasting 22(4). Reviews forecast accuracy measures, discusses their shortcomings, and proposes the mean absolute scaled error (MASE).
- `taieb2012review` -- Expert Systems with Applications 39(8). Reviews and compares recursive, direct, DirRec and multi-output strategies for multi-step-ahead forecasting on the 111 NN5 competition series; multiple-output strategies perform best and deseasonalisation uniformly improves accuracy.

## Optimisation, MPC and decisions under uncertainty

- `rawlings2017model` -- Nob Hill Publishing, 2nd ed. (ISBN 9780975937730). Standard textbook on model predictive control theory, computation and design.
- `mayne2000constrained` -- Automatica 36(6). Survey of stability and optimality principles for constrained (linear and nonlinear) model predictive control.
- `powell2019unified` -- EJOR 275(3). Proposes a unified modelling framework for sequential decisions under uncertainty that optimises over policies, identifying four classes of policies (including lookahead policies such as MPC).
- `bertsimas2004price` -- Operations Research 52(1). Robust linear optimisation with a budget of uncertainty that controls the trade-off between robustness and conservatism ("price of robustness").
- `bental2000robust` -- Mathematical Programming 88(3). Shows that NETLIB LP solutions can be badly infeasible under small data perturbations and applies robust optimisation to obtain solutions immunised against uncertainty.
- `smith2006optimizer` -- Management Science 52(3). The optimizer's curse: choosing the alternative with the highest estimated value leads to expected post-decision disappointment even with unbiased estimates; proposes Bayesian correction.
- `ahuja1993network` -- Prentice Hall book (ISBN 013617549X). Standard reference on network flow theory and algorithms, including minimum-cost flow.
- `orlin1997polynomial` -- Mathematical Programming 78(2). Gives a polynomial-time primal network simplex algorithm for the minimum-cost flow problem via a premultiplier method with cost scaling.
- `ortools` -- Google OR-Tools (software, version 9.15.6755 as used in the code). Open-source operations research suite; its min-cost-flow solver is used for the scheduler.

## Sustainable operations context

- `gahm2016energy` -- EJOR 248(3). Reviews energy-efficient scheduling in manufacturing and proposes a research framework structured by energetic coverage, energy supply and energy demand.
- `asadpour2022green` -- Sustainable Operations and Computers 3. Green bi-objective model for identical parallel-machine scheduling with job splitting that trades off tardy jobs against energy/environmental objectives.
- `sharma2024efficient` -- Sustainable Operations and Computers 5. Hybrid spotted hyena optimiser and artificial neural network (SHO-ANN) for virtual machine resource allocation in cloud environments.
- `wu2022review` -- Sustainable Operations and Computers 3. Review of theoretical research and practical progress towards carbon neutrality, including pathways and sectoral studies.

## Not included (could not verify or out of scope)

- IEA "Electricity 2024" report: the iea.org page could not be retrieved (blocked by a bot check), so it was not verified and is excluded.
- NESO/National Grid ESO "Carbon Intensity Forecast Methodology" (Bruce, Ruff et al.): the GitHub-hosted PDF could not be retrieved in this environment; excluded. The API itself is cited as `nesoapi`.
- Xu & Xie ICML 2021 EnbPI paper: not in Crossref; the extended journal version `xu2023conformal` (IEEE TPAMI, verified) is cited instead.
- Pages for `friedman2001greedy`, `romano2019conformalized` and `ke2017lightgbm` are omitted because the checked sources did not list them.
- OR-Tools author attribution (Perron and Furnon) could not be confirmed from the accessible pages, so `ortools` is attributed to Google.
