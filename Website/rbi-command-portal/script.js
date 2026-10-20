const crises = [
  {
    title: "China Yuan Devaluation",
    dates: "2015-08-11 to 2015-09-15",
    severity: 35,
    affected: "Axis, HDFC, Kotak, IndusInd",
    cue: "FX-exposed lenders on watch",
    alert: "Cross-border stress transmission",
    context:
      "Global market weakness following the yuan devaluation placed pressure on Indian banks with foreign-exchange sensitivity and external market exposure.",
  },
  {
    title: "Demonetization",
    dates: "2016-11-08 to 2016-12-31",
    severity: 55,
    affected: "All 33 banks",
    cue: "Cash pressure and branch surge",
    alert: "System-wide liquidity operations",
    context:
      "The withdrawal of INR 500 and INR 1000 notes produced an operational surge, deposit realignment, and system-wide liquidity management pressure across the banking network.",
  },
  {
    title: "IL&FS Liquidity Crisis",
    dates: "2018-09-21 to 2018-11-30",
    severity: 45,
    affected: "PSU banks and IDBI-linked exposure",
    cue: "NBFC-linked transmission stress",
    alert: "Liquidity freeze watch",
    context:
      "The IL&FS default triggered funding stress in the NBFC ecosystem, with contagion concern flowing into banks holding direct or indirect exposure to the sector.",
  },
  {
    title: "Yes Bank Stress",
    dates: "2019-03-05 to 2020-03-04",
    severity: 75,
    affected: "Yes Bank, RBL, IndusInd",
    cue: "Long-horizon supervisory stress",
    alert: "Institution-led network instability",
    context:
      "The master plan marks this as a sustained stress period driven by deteriorating asset quality, rating pressure, and escalating concern around Yes Bank's resilience.",
  },
  {
    title: "Yes Bank Moratorium",
    dates: "2020-03-05 to 2020-03-13",
    severity: 82,
    affected: "Yes Bank and rescue-linked peers",
    cue: "Moratorium and SBI-led rescue response",
    alert: "Acute intervention window",
    context:
      "The moratorium period represented an immediate institutional shock, including withdrawal restrictions and an emergency rescue structure to contain broader instability.",
  },
  {
    title: "COVID-19 Pandemic",
    dates: "2020-03-15 to 2020-06-30",
    severity: 92,
    affected: "All 33 banks",
    cue: "Forced vertical SRI spike",
    alert: "System-wide alert",
    context:
      "National lockdown, emergency rate cuts, loan moratorium measures, and broad uncertainty created the strongest systemic stress event in the brief, with the SRI forced above 0.92 during the peak phase.",
  },
  {
    title: "Second COVID Wave",
    dates: "2021-04-01 to 2021-05-31",
    severity: 58,
    affected: "All 33 banks",
    cue: "Localized lockdown balance-sheet pressure",
    alert: "Renewed health-linked strain",
    context:
      "The second wave brought renewed local restrictions, credit quality concern, and economic contraction pressure, though below the first COVID shock in systemic magnitude.",
  },
  {
    title: "RBI Rate Hike Stress",
    dates: "2022-05-04 to 2022-09-30",
    severity: 52,
    affected: "IDFC FIRST, IndusInd, Kotak, SFBs",
    cue: "Rate-sensitive balance sheets under review",
    alert: "Policy tightening watch",
    context:
      "Rapid RBI rate hikes stressed rate-sensitive institutions, especially smaller lenders and banks more exposed to funding cost repricing and borrower strain.",
  },
  {
    title: "SVB Global Contagion",
    dates: "2023-03-10 to 2023-04-15",
    severity: 42,
    affected: "AU, Equitas, Ujjivan, CSB, City Union",
    cue: "Deposit confidence sensitivity",
    alert: "Imported sentiment stress",
    context:
      "The collapse of Silicon Valley Bank drove global banking anxiety, with the brief highlighting confidence-sensitive smaller institutions as the most watch-worthy in India.",
  },
];

const banks = [
  {
    id: "sbi",
    name: "State Bank of India (SBI)",
    classification: "Core Bank",
    role: "Anchor node",
    type: "Core / PSU",
    cue: "System-wide relevance",
    filters: ["all", "core", "psu"],
    reason:
      "Largest Indian bank and a domestic systemically important institution with direct national oversight significance.",
  },
  {
    id: "hdfc",
    name: "HDFC Bank",
    classification: "Core Bank",
    role: "Anchor node",
    type: "Core / Private",
    cue: "Large-cap private system anchor",
    filters: ["all", "core", "private"],
    reason:
      "Largest private-sector bank in the brief and one of the core institutions around which the contagion network is organised.",
  },
  {
    id: "icici",
    name: "ICICI Bank",
    classification: "Core Bank",
    role: "Anchor node",
    type: "Core / Private",
    cue: "Cross-market transmission relevance",
    filters: ["all", "core", "private"],
    reason:
      "Large private-sector lender with global operations and central importance to the core-periphery network structure.",
  },
  {
    id: "axis",
    name: "Axis Bank",
    classification: "Core Bank",
    role: "Core connector",
    type: "Core / Private",
    cue: "High interbank exposure",
    filters: ["all", "core", "private"],
    reason:
      "The brief positions Axis Bank as a core institution with meaningful interbank exposure and contagion relevance.",
  },
  {
    id: "kotak",
    name: "Kotak Mahindra Bank",
    classification: "Core Bank",
    role: "Core connector",
    type: "Core / Private",
    cue: "High network centrality",
    filters: ["all", "core", "private"],
    reason:
      "Systemically important, highly connected, and explicitly identified as one of the core banks in the monitoring design.",
  },
  {
    id: "au",
    name: "AU Small Finance Bank",
    classification: "Periphery Bank",
    role: "Specialized risk node",
    type: "Periphery / Small Finance",
    cue: "Rate-sensitive niche lender",
    filters: ["all", "sfb", "private"],
    reason:
      "A niche small finance bank included because the brief expects sharper sensitivity during rate and confidence shocks.",
  },
  {
    id: "bandhan",
    name: "Bandhan Bank",
    classification: "Periphery Bank",
    role: "Specialized risk node",
    type: "Periphery / Small Finance lineage",
    cue: "Rural and microfinance sensitivity",
    filters: ["all", "sfb", "private"],
    reason:
      "Included for its microfinance-converted profile and exposure to rural and economically sensitive borrower segments.",
  },
  {
    id: "bob",
    name: "Bank of Baroda",
    classification: "Periphery Bank",
    role: "PSU transmission node",
    type: "Periphery / PSU",
    cue: "Post-merger scale",
    filters: ["all", "psu"],
    reason:
      "A large public sector bank with post-merger scale, important for understanding state-backed system-wide spillovers.",
  },
  {
    id: "boi",
    name: "Bank of India",
    classification: "Periphery Bank",
    role: "PSU transmission node",
    type: "Periphery / PSU",
    cue: "Historical asset-quality watch",
    filters: ["all", "psu"],
    reason:
      "The brief flags historical NPA concerns, making it relevant in stress propagation and supervisory interpretation.",
  },
  {
    id: "bom",
    name: "Bank of Maharashtra",
    classification: "Periphery Bank",
    role: "Regional PSU node",
    type: "Periphery / PSU",
    cue: "Regional focus",
    filters: ["all", "psu", "regional"],
    reason:
      "A public sector lender with regional concentration, valuable for reading targeted rather than purely system-wide pressure.",
  },
  {
    id: "canara",
    name: "Canara Bank",
    classification: "Periphery Bank",
    role: "PSU transmission node",
    type: "Periphery / PSU",
    cue: "Post-Syndicate merger scale",
    filters: ["all", "psu"],
    reason:
      "A major public sector institution whose post-merger footprint makes it significant in broad banking-system surveillance.",
  },
  {
    id: "central",
    name: "Central Bank of India",
    classification: "Periphery Bank",
    role: "PSU watch node",
    type: "Periphery / PSU",
    cue: "PCA history",
    filters: ["all", "psu"],
    reason:
      "Included for legacy supervisory concerns and historical Prompt Corrective Action relevance.",
  },
  {
    id: "cub",
    name: "City Union Bank",
    classification: "Periphery Bank",
    role: "Regional private node",
    type: "Periphery / Regional Private",
    cue: "Trade finance focus",
    filters: ["all", "private", "regional"],
    reason:
      "A south-focused private bank with trade finance links, useful for capturing regional and sector-specific transmission patterns.",
  },
  {
    id: "csb",
    name: "CSB Bank",
    classification: "Periphery Bank",
    role: "Regional private node",
    type: "Periphery / Regional Private",
    cue: "Smaller private balance sheet",
    filters: ["all", "private", "regional"],
    reason:
      "A smaller private lender included in the periphery set where funding confidence and localized stress can matter quickly.",
  },
  {
    id: "equitas",
    name: "Equitas Small Finance Bank",
    classification: "Periphery Bank",
    role: "Specialized risk node",
    type: "Periphery / Small Finance",
    cue: "High-growth rate sensitivity",
    filters: ["all", "sfb", "private"],
    reason:
      "The brief treats Equitas as a rate-sensitive growth lender, especially relevant in policy tightening and deposit confidence episodes.",
  },
  {
    id: "federal",
    name: "Federal Bank",
    classification: "Periphery Bank",
    role: "Private transmission node",
    type: "Periphery / Private",
    cue: "Kerala-rooted private franchise",
    filters: ["all", "private", "regional"],
    reason:
      "A well-known private lender with regional depth, included to reflect how concentrated market bases behave under stress.",
  },
  {
    id: "idbi",
    name: "IDBI Bank",
    classification: "Periphery Bank",
    role: "PSU-linked node",
    type: "Periphery / PSU-backed",
    cue: "LIC-backed transition profile",
    filters: ["all", "psu"],
    reason:
      "A transitional institution with public-sector backing and legacy infrastructure-finance associations, important for contagion realism.",
  },
  {
    id: "idfc-first",
    name: "IDFC FIRST Bank",
    classification: "Periphery Bank",
    role: "Private sensitivity node",
    type: "Periphery / Private",
    cue: "Rate and funding sensitivity",
    filters: ["all", "private"],
    reason:
      "The rate-hike portion of the brief specifically highlights IDFC FIRST as a bank worth watching for funding-cost pressure.",
  },
  {
    id: "indian-bank",
    name: "Indian Bank",
    classification: "Periphery Bank",
    role: "PSU transmission node",
    type: "Periphery / PSU",
    cue: "Post-merger public sector scale",
    filters: ["all", "psu"],
    reason:
      "A merged public sector institution that helps represent how broader state banking networks absorb and transmit pressure.",
  },
  {
    id: "iob",
    name: "Indian Overseas Bank (IOB)",
    classification: "Periphery Bank",
    role: "PSU watch node",
    type: "Periphery / PSU",
    cue: "Historic PCA relevance",
    filters: ["all", "psu"],
    reason:
      "Included because historical corrective-action context makes it useful in a serious supervisory watch framework.",
  },
  {
    id: "indusind",
    name: "IndusInd Bank",
    classification: "Periphery Bank",
    role: "Private contagion node",
    type: "Periphery / Private",
    cue: "Repeated presence in stress windows",
    filters: ["all", "private"],
    reason:
      "Appears repeatedly in the brief’s crisis mappings, making it a meaningful non-core bank for contagion narratives.",
  },
  {
    id: "jk-bank",
    name: "J&K Bank",
    classification: "Periphery Bank",
    role: "Regional watch node",
    type: "Periphery / Regional",
    cue: "Territory-focused exposure",
    filters: ["all", "regional"],
    reason:
      "A territory-focused institution that introduces geographic specificity into the otherwise system-wide watch universe.",
  },
  {
    id: "karnataka",
    name: "Karnataka Bank",
    classification: "Periphery Bank",
    role: "Regional private node",
    type: "Periphery / Regional Private",
    cue: "Southern regional lens",
    filters: ["all", "private", "regional"],
    reason:
      "A regional private bank that helps the shell reflect the full spread of India’s institutional banking landscape.",
  },
  {
    id: "karur",
    name: "Karur Vysya Bank",
    classification: "Periphery Bank",
    role: "Regional private node",
    type: "Periphery / Regional Private",
    cue: "Trade-linked franchise",
    filters: ["all", "private", "regional"],
    reason:
      "Included for its regional strength and trade-oriented posture, which adds realistic texture to the network.",
  },
  {
    id: "psb",
    name: "Punjab & Sind Bank",
    classification: "Periphery Bank",
    role: "PSU watch node",
    type: "Periphery / PSU",
    cue: "Smaller public-sector footprint",
    filters: ["all", "psu"],
    reason:
      "A smaller public sector lender that rounds out the supervisory picture beyond only the largest PSU institutions.",
  },
  {
    id: "pnb",
    name: "Punjab National Bank (PNB)",
    classification: "Periphery Bank",
    role: "PSU transmission node",
    type: "Periphery / PSU",
    cue: "High public importance",
    filters: ["all", "psu"],
    reason:
      "A major public sector bank with prominent historical control and governance lessons relevant to systemic credibility.",
  },
  {
    id: "rbl",
    name: "RBL Bank",
    classification: "Periphery Bank",
    role: "Private contagion node",
    type: "Periphery / Private",
    cue: "Linked to Yes Bank stress narrative",
    filters: ["all", "private"],
    reason:
      "The brief repeatedly associates RBL with the Yes Bank stress arc, making it a natural watch candidate in contagion scenarios.",
  },
  {
    id: "sib",
    name: "South Indian Bank",
    classification: "Periphery Bank",
    role: "Regional private node",
    type: "Periphery / Regional Private",
    cue: "Southern private lender exposure",
    filters: ["all", "private", "regional"],
    reason:
      "A regional private bank used to capture how localized franchises may respond differently to stress episodes.",
  },
  {
    id: "tmb",
    name: "Tamilnad Mercantile Bank (TMB)",
    classification: "Periphery Bank",
    role: "Regional private node",
    type: "Periphery / Regional Private",
    cue: "Tamil Nadu regional depth",
    filters: ["all", "private", "regional"],
    reason:
      "Its regional profile helps the interface represent institutional diversity beyond the large-cap banks.",
  },
  {
    id: "uco",
    name: "UCO Bank",
    classification: "Periphery Bank",
    role: "PSU watch node",
    type: "Periphery / PSU",
    cue: "Legacy PCA relevance",
    filters: ["all", "psu"],
    reason:
      "A public sector bank with historical supervisory significance that suits a serious financial monitoring environment.",
  },
  {
    id: "ujjivan",
    name: "Ujjivan Small Finance Bank",
    classification: "Periphery Bank",
    role: "Specialized risk node",
    type: "Periphery / Small Finance",
    cue: "Confidence-sensitive funding profile",
    filters: ["all", "sfb", "private"],
    reason:
      "The brief highlights Ujjivan in the global contagion window, especially where depositor confidence matters.",
  },
  {
    id: "union",
    name: "Union Bank of India",
    classification: "Periphery Bank",
    role: "PSU transmission node",
    type: "Periphery / PSU",
    cue: "Large merged public-sector platform",
    filters: ["all", "psu"],
    reason:
      "A large PSU institution whose size makes it useful for understanding broad state-bank system responses.",
  },
  {
    id: "yes-bank",
    name: "Yes Bank",
    classification: "Periphery Bank",
    role: "Distress reference node",
    type: "Periphery / Private",
    cue: "Moratorium-era critical watch",
    filters: ["all", "private"],
    reason:
      "A central stress case in the brief, explicitly tied to long-horizon vulnerability, moratorium shock, rescue action, and explainability outputs.",
  },
];

const dockLayouts = {
  embedded: {
    kicker: "Embedded module",
    title: "Place the live dashboard inside this secure presentation frame.",
    description:
      "Best for keeping the institutional landing experience visible while your actual analytics module sits in a controlled content well below the fold.",
    points: [
      "Keep this shell as the public-facing supervisory layer.",
      "Mount your future app inside a dedicated content container or iframe.",
      "Preserve the navigation, trust signals, and institutional framing above.",
    ],
    snippet: `<section id="project-surface">\n  Add your live dashboard, charts, APIs, or model interface here.\n</section>`,
  },
  sidecar: {
    kicker: "Sidecar workspace",
    title: "Run your future dashboard beside this portal while keeping the shell separate.",
    description:
      "Useful when the actual project needs its own routing, runtime, or frontend stack but you still want this shell to present context, trust, and executive framing.",
    points: [
      "Keep this portal static and independent as the oversight entry point.",
      "Link the live project in a second panel, route, subdomain, or authenticated workspace.",
      "Use the shell for brief summaries while the heavy analytics live elsewhere.",
    ],
    snippet: `<div class="portal-grid">\n  <aside>Institutional shell</aside>\n  <main>Independent live project</main>\n</div>`,
  },
  takeover: {
    kicker: "Full takeover",
    title: "Let the real application inherit the entire central surface when you are ready.",
    description:
      "Ideal if the final build becomes mature enough to replace the placeholder area entirely while still keeping the same visual language and header structure.",
    points: [
      "Retain the header, background watermark, and institutional palette.",
      "Swap the placeholder section for the production dashboard root node.",
      "Use this shell as the design system basis for the finished platform.",
    ],
    snippet: `<main id="app-root">\n  <!-- Your production analytics application takes over here -->\n</main>`,
  },
};

const crisisSlider = document.getElementById("crisisSlider");
const prevCrisisButton = document.getElementById("prevCrisis");
const nextCrisisButton = document.getElementById("nextCrisis");
const crisisList = document.getElementById("crisisList");
const crisisIndex = document.getElementById("crisisIndex");
const watchLevel = document.getElementById("watchLevel");
const heroEventName = document.getElementById("heroEventName");
const heroSeverity = document.getElementById("heroSeverity");
const heroAffected = document.getElementById("heroAffected");
const heroWindow = document.getElementById("heroWindow");
const heroContext = document.getElementById("heroContext");
const crisisAlert = document.getElementById("crisisAlert");
const crisisTitle = document.getElementById("crisisTitle");
const crisisDates = document.getElementById("crisisDates");
const crisisContext = document.getElementById("crisisContext");
const crisisSeverity = document.getElementById("crisisSeverity");
const crisisAffected = document.getElementById("crisisAffected");
const crisisCue = document.getElementById("crisisCue");

const bankGrid = document.getElementById("bankGrid");
const filterRow = document.getElementById("filterRow");
const bankSearch = document.getElementById("bankSearch");
const bankClassification = document.getElementById("bankClassification");
const bankName = document.getElementById("bankName");
const bankReason = document.getElementById("bankReason");
const bankRole = document.getElementById("bankRole");
const bankType = document.getElementById("bankType");
const bankCue = document.getElementById("bankCue");

const dockModeSwitch = null;

let currentCrisisIndex = 5;
let currentFilter = "all";
let currentSearch = "";
let activeBankId = "sbi";

function getWatchProfile(severity) {
  if (severity >= 75) {
    return { label: "Critical Watch", state: "critical" };
  }

  if (severity >= 50) {
    return { label: "Elevated Watch", state: "elevated" };
  }

  return { label: "Focused Watch", state: "focused" };
}

function updateCrisis(index) {
  currentCrisisIndex = index;
  const crisis = crises[index];
  const profile = getWatchProfile(crisis.severity);

  crisisIndex.textContent = `${String(index + 1).padStart(2, "0")} / ${String(
    crises.length
  ).padStart(2, "0")}`;
  crisisSlider.value = index;

  watchLevel.textContent = profile.label;
  watchLevel.dataset.state = profile.state;

  heroEventName.textContent = crisis.title;
  heroSeverity.textContent = `${crisis.severity}%`;
  heroAffected.textContent = crisis.affected;
  heroWindow.textContent = crisis.dates;
  heroContext.textContent = crisis.context;

  crisisAlert.textContent = crisis.alert;
  crisisTitle.textContent = crisis.title;
  crisisDates.textContent = crisis.dates;
  crisisContext.textContent = crisis.context;
  crisisSeverity.textContent = `${crisis.severity}%`;
  crisisAffected.textContent = crisis.affected;
  crisisCue.textContent = crisis.cue;

  [...crisisList.children].forEach((item, itemIndex) => {
    item.classList.toggle("active", itemIndex === index);
    item.setAttribute("aria-pressed", itemIndex === index ? "true" : "false");
  });
}

function renderCrises() {
  crisisList.innerHTML = "";

  crises.forEach((crisis, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "timeline-item";
    button.innerHTML = `
      <h4>${crisis.title}</h4>
      <p>${crisis.dates}</p>
      <p>${crisis.cue}</p>
    `;
    button.addEventListener("click", () => updateCrisis(index));
    crisisList.appendChild(button);
  });

  updateCrisis(currentCrisisIndex);
}

function getFilteredBanks() {
  return banks.filter((bank) => {
    const matchesFilter = currentFilter === "all" || bank.filters.includes(currentFilter);
    const matchesSearch =
      currentSearch.length === 0 ||
      bank.name.toLowerCase().includes(currentSearch.toLowerCase());
    return matchesFilter && matchesSearch;
  });
}

function updateBankDetail(bank) {
  bankClassification.textContent = bank.classification;
  bankName.textContent = bank.name;
  bankReason.textContent = bank.reason;
  bankRole.textContent = bank.role;
  bankType.textContent = bank.type;
  bankCue.textContent = bank.cue;
}

function renderBanks() {
  const visibleBanks = getFilteredBanks();
  bankGrid.innerHTML = "";

  if (!visibleBanks.some((bank) => bank.id === activeBankId)) {
    activeBankId = visibleBanks[0]?.id || "";
  }

  if (visibleBanks.length === 0) {
    bankGrid.innerHTML = `<div class="bank-chip"><span class="chip-class">No result</span><span class="chip-name">No bank matches your current filter.</span></div>`;
    bankClassification.textContent = "Selection unavailable";
    bankName.textContent = "No matching institution";
    bankReason.textContent =
      "Try a broader filter or clear the search field to restore the full coverage list.";
    bankRole.textContent = "Not available";
    bankType.textContent = "Not available";
    bankCue.textContent = "Adjust filters";
    return;
  }

  visibleBanks.forEach((bank) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "bank-chip";
    button.innerHTML = `
      <span class="chip-class">${bank.classification}</span>
      <span class="chip-name">${bank.name}</span>
    `;
    button.classList.toggle("active", bank.id === activeBankId);
    button.setAttribute("aria-pressed", bank.id === activeBankId ? "true" : "false");
    button.addEventListener("click", () => {
      activeBankId = bank.id;
      renderBanks();
    });
    bankGrid.appendChild(button);
  });

  const activeBank = visibleBanks.find((bank) => bank.id === activeBankId) || visibleBanks[0];
  activeBankId = activeBank.id;
  updateBankDetail(activeBank);
}

function updateFilterButtons() {
  [...filterRow.querySelectorAll(".filter-button")].forEach((button) => {
    const isActive = button.dataset.filter === currentFilter;
    button.classList.toggle("active", isActive);
    button.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function activateReveal() {
  const revealItems = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.18,
    }
  );

  revealItems.forEach((item) => observer.observe(item));
}

renderCrises();
renderBanks();
updateFilterButtons();
activateReveal();

crisisSlider.addEventListener("input", (event) => {
  updateCrisis(Number(event.target.value));
});

prevCrisisButton.addEventListener("click", () => {
  updateCrisis((currentCrisisIndex - 1 + crises.length) % crises.length);
});

nextCrisisButton.addEventListener("click", () => {
  updateCrisis((currentCrisisIndex + 1) % crises.length);
});

filterRow.addEventListener("click", (event) => {
  const button = event.target.closest(".filter-button");
  if (!button) {
    return;
  }

  currentFilter = button.dataset.filter;
  updateFilterButtons();
  renderBanks();
});

bankSearch.addEventListener("input", (event) => {
  currentSearch = event.target.value.trim();
  renderBanks();
});
