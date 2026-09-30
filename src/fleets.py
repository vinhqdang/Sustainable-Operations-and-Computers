"""Fleet definitions: four candidate sites per fleet (zones ordered from the cleanest
to the dirtiest mean intensity, which is also the order of capacity and job-origin
shares: the cleanest site is the smallest, as in the Great Britain study).

tz: UTC offset of local standard time in hours (diurnal workload and interactive load
follow local time). pue: uniform 1.20 outside Great Britain.
Zone codes: Great Britain = NESO region ids; Europe = Energy-Charts country codes;
United States = EIA balancing authorities; Brazil = ONS subsystems; NZ = national."""
FLEETS = {
    "GB": dict(label="Great Britain (4 NESO regions)", zones=[2, 3, 8, 13],
               names=["South Scotland", "North West England", "West Midlands", "London"],
               tz=[0, 0, 0, 0], pue=[1.15, 1.20, 1.25, 1.30]),
    "EU": dict(label="Europe (FR, ES, DE, PL)", zones=["FR", "ES", "DE", "PL"],
               names=["France", "Spain", "Germany", "Poland"], tz=[1, 1, 1, 1], pue=[1.2] * 4),
    "US": dict(label="United States (BPAT, CISO, ERCO, PJM)", zones=["BPAT", "CISO", "ERCO", "PJM"],
               names=["BPA (Pacific NW)", "California", "Texas", "PJM (Mid-Atlantic)"],
               tz=[-8, -8, -6, -5], pue=[1.2] * 4),
    "BR": dict(label="Brazil (NE, SE/CO, S, N)", zones=["BR_NE", "BR_SE", "BR_S", "BR_N"],
               names=["Nordeste", "Sudeste/CO", "Sul", "Norte"], tz=[-3, -3, -3, -3], pue=[1.2] * 4),
    "GLOBAL": dict(label="Four continents (FR, NZ, BR-S, CISO)", zones=["FR", "NZ", "BR_S", "CISO"],
                   names=["France", "New Zealand", "Brazil South", "California"],
                   tz=[1, 12, -3, -8], pue=[1.2] * 4),
}
CAP = [300.0, 400.0, 400.0, 600.0]          # batch servers, cleanest site first
ORIGIN_P = [0.10, 0.20, 0.25, 0.45]          # job-origin shares, cleanest site first
