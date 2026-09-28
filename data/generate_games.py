"""
Synthetic multi-sport games dataset generator.
Fictional event: "Meridian Games 2028". All people/results are synthetic.
Outputs 5 CSVs: Athletes, Coaches, EntriesGender, Medals, Teams
"""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
OUT = Path("/mnt/user-data/outputs/meridian_games")
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- config
N_ATHLETES = 110_000          # roster across the whole games history
N_EDITIONS = 1                # single event, but athletes can enter many events

# country: (weight for delegation size, tier for medal strength 0-1)
COUNTRIES = {
    "United States": (10.0, 0.95), "China": (9.0, 0.93), "Japan": (7.0, 0.80),
    "Great Britain": (6.0, 0.82), "Germany": (6.0, 0.78), "France": (5.5, 0.76),
    "Australia": (5.0, 0.75), "Italy": (4.5, 0.70), "Netherlands": (4.0, 0.72),
    "Canada": (4.0, 0.65), "Brazil": (4.0, 0.60), "South Korea": (3.8, 0.68),
    "Spain": (3.5, 0.58), "Russia": (3.5, 0.74), "India": (3.5, 0.40),
    "Kenya": (2.0, 0.55), "Jamaica": (1.8, 0.62), "Ethiopia": (1.6, 0.50),
    "New Zealand": (2.0, 0.60), "Sweden": (2.2, 0.55), "Norway": (2.0, 0.60),
    "Poland": (2.5, 0.50), "Hungary": (1.8, 0.55), "Ukraine": (2.2, 0.52),
    "Argentina": (2.5, 0.45), "Mexico": (2.5, 0.40), "Egypt": (1.8, 0.38),
    "Nigeria": (2.0, 0.38), "South Africa": (2.2, 0.48), "Turkey": (2.4, 0.45),
    "Iran": (1.8, 0.42), "Thailand": (1.6, 0.35), "Vietnam": (1.2, 0.28),
    "Indonesia": (1.6, 0.32), "Colombia": (1.6, 0.35), "Chile": (1.2, 0.30),
    "Cuba": (1.4, 0.50), "Croatia": (1.4, 0.45), "Switzerland": (2.0, 0.55),
    "Denmark": (1.6, 0.50), "Czechia": (1.6, 0.48), "Greece": (1.5, 0.40),
    "Portugal": (1.5, 0.38), "Ireland": (1.3, 0.40), "Belgium": (1.6, 0.45),
    "Morocco": (1.3, 0.32), "Uzbekistan": (1.3, 0.38), "Kazakhstan": (1.4, 0.40),
    "Saudi Arabia": (1.0, 0.22), "Fiji": (0.5, 0.25), "Mongolia": (0.6, 0.30),
    "Costa Rica": (0.6, 0.20), "Uganda": (0.7, 0.30), "Jordan": (0.5, 0.18),
    "Lithuania": (0.9, 0.35), "Serbia": (1.3, 0.48), "Slovenia": (1.0, 0.42),
}
COUNTRY_NAMES = list(COUNTRIES)
C_WEIGHTS = np.array([COUNTRIES[c][0] for c in COUNTRY_NAMES])
C_WEIGHTS = C_WEIGHTS / C_WEIGHTS.sum()
C_TIER = {c: COUNTRIES[c][1] for c in COUNTRY_NAMES}

# discipline: (weight, team_event?, gender_mix: (men, women, mixed), n_events, avg_team_size)
DISCIPLINES = {
    "Athletics":        (14.0, False, (0.50, 0.50), 48, 1),
    "Swimming":         (9.0,  False, (0.50, 0.50), 37, 1),
    "Gymnastics":       (4.0,  False, (0.45, 0.55), 18, 1),
    "Cycling":          (4.0,  False, (0.58, 0.42), 20, 1),
    "Rowing":           (3.0,  True,  (0.52, 0.48), 14, 6),
    "Football":         (5.0,  True,  (0.60, 0.40), 2, 18),
    "Basketball":       (3.5,  True,  (0.55, 0.45), 4, 12),
    "Volleyball":       (3.0,  True,  (0.50, 0.50), 4, 12),
    "Hockey":           (2.5,  True,  (0.55, 0.45), 2, 16),
    "Handball":         (2.5,  True,  (0.50, 0.50), 2, 14),
    "Judo":             (3.0,  False, (0.55, 0.45), 14, 1),
    "Boxing":           (2.5,  False, (0.70, 0.30), 13, 1),
    "Wrestling":        (2.5,  False, (0.65, 0.35), 18, 1),
    "Weightlifting":    (2.0,  False, (0.55, 0.45), 14, 1),
    "Tennis":           (2.0,  False, (0.50, 0.50), 5, 1),
    "Badminton":        (2.0,  False, (0.50, 0.50), 5, 1),
    "Table Tennis":     (1.8,  False, (0.50, 0.50), 5, 1),
    "Fencing":          (2.0,  False, (0.50, 0.50), 12, 1),
    "Archery":          (1.6,  False, (0.50, 0.50), 5, 1),
    "Shooting":         (2.0,  False, (0.55, 0.45), 15, 1),
    "Diving":           (1.5,  False, (0.45, 0.55), 8, 1),
    "Sailing":          (1.6,  False, (0.55, 0.45), 10, 1),
    "Equestrian":       (1.4,  False, (0.40, 0.60), 6, 1),
    "Triathlon":        (1.2,  False, (0.50, 0.50), 3, 1),
    "Rugby Sevens":     (1.8,  True,  (0.55, 0.45), 2, 12),
    "Water Polo":       (1.5,  True,  (0.55, 0.45), 2, 13),
    "Canoe Sprint":     (1.6,  False, (0.58, 0.42), 10, 1),
    "Skateboarding":    (1.0,  False, (0.55, 0.45), 4, 1),
    "Climbing":         (0.9,  False, (0.50, 0.50), 4, 1),
    "Surfing":          (0.8,  False, (0.50, 0.50), 2, 1),
    "Taekwondo":        (1.6,  False, (0.50, 0.50), 8, 1),
    "Modern Pentathlon":(0.7,  False, (0.50, 0.50), 2, 1),
}
D_NAMES = list(DISCIPLINES)
D_WEIGHTS = np.array([DISCIPLINES[d][0] for d in D_NAMES])
D_WEIGHTS = D_WEIGHTS / D_WEIGHTS.sum()

# ---------------------------------------------------------------- name pools
# Culturally plausible given/family names per country group. Purely synthetic
# combinations; any real-person match is coincidental.
NAME_POOLS = {
    # pool -> {"M": [...], "F": [...], "L": [...]}  (male first, female first, last)
    "anglo": {
        "M": ["James","Oliver","Liam","Noah","Ethan","Lucas","Mason","Logan","Jack","Harry","Daniel","Matthew","Ryan","Tyler","Connor","Jordan","Callum","Owen","Dylan","Sean"],
        "F": ["Emma","Olivia","Ava","Sophia","Isabella","Mia","Charlotte","Amelia","Grace","Chloe","Hannah","Ella","Lily","Megan","Erin","Holly","Zoe","Abigail","Molly","Freya"],
        "L": ["Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Wilson","Taylor","Clark","Walker","Hall","Young","King","Wright","Scott","Green","Baker","Adams","Nelson","Carter","Mitchell","Turner","Phillips","Campbell","Parker","Evans","Edwards","Collins","Stewart"]},
    "germanic": {
        "M": ["Lukas","Felix","Jonas","Leon","Maximilian","Paul","Finn","Tim","Jan","Niklas","Erik","Nils","Sven","Mats","Bram","Daan","Lars","Anders","Henrik","Oskar"],
        "F": ["Anna","Lena","Marie","Sophie","Laura","Julia","Hannah","Lea","Emilia","Clara","Sanne","Femke","Ingrid","Astrid","Freja","Signe","Maja","Linnea","Elin","Saga"],
        "L": ["Muller","Schmidt","Schneider","Fischer","Weber","Meyer","Wagner","Becker","Hoffmann","Koch","Richter","Klein","Wolf","Neumann","Schwarz","Braun","Krause","Lange","Vogel","Jansen","De Vries","Bakker","Visser","Andersson","Lindqvist","Nilsson","Hansen","Berg","Larsen","Dijkstra"]},
    "romance": {
        "M": ["Mateo","Hugo","Lucas","Leo","Gabriel","Louis","Luca","Marco","Andrea","Diego","Pablo","Javier","Carlos","Antoine","Julien","Matteo","Alessandro","Joao","Tiago","Rui"],
        "F": ["Sofia","Camille","Lea","Chiara","Giulia","Lucia","Elena","Carmen","Ines","Manon","Marta","Paula","Alba","Valentina","Beatriz","Ana","Clara","Julia","Martina","Noemi"],
        "L": ["Garcia","Martinez","Lopez","Sanchez","Fernandez","Rossi","Russo","Ferrari","Esposito","Bianchi","Martin","Bernard","Dubois","Moreau","Laurent","Simon","Silva","Santos","Costa","Oliveira","Pereira","Rodrigues","Gomez","Diaz","Torres","Ramirez","Flores","Romero","Vargas","Castillo"]},
    "slavic": {
        "M": ["Ivan","Dmitri","Alexei","Nikolai","Pavel","Marek","Tomas","Milan","Luka","Nikola","Andrei","Viktor","Bogdan","Jakub","Piotr","Mateusz","Stefan","Goran","Zoran","Damir"],
        "F": ["Anastasia","Olga","Natalia","Katarina","Marta","Ivana","Jana","Petra","Ana","Mila","Irina","Svetlana","Agnieszka","Zofia","Milica","Tatiana","Vesna","Dragana","Nina","Eva"],
        "L": ["Ivanov","Petrov","Smirnov","Kuznetsov","Popov","Novak","Horvat","Kovac","Kowalski","Nowak","Wisniewski","Jovanovic","Petrovic","Nikolic","Kovalenko","Shevchenko","Bondarenko","Marin","Toth","Varga","Volkov","Sokolov","Morozov","Lebedev","Kaminski","Zielinski","Babic","Simic","Kolar","Vidmar"]},
    "east_asian": {
        "M": ["Hiroshi","Kenji","Takumi","Ryota","Daiki","Haruto","Yuto","Sota","Minato","Ren","Wei","Jun","Lei","Hao","Ming","Jian","Minjun","Seojun","Jiho","Hyun"],
        "F": ["Yuki","Sakura","Hana","Aoi","Mei","Rin","Akari","Yui","Miyu","Nanami","Xin","Yan","Fang","Jing","Ling","Jisoo","Minseo","Seoyeon","Haeun","Yerin"],
        "L": ["Sato","Suzuki","Takahashi","Tanaka","Watanabe","Ito","Yamamoto","Nakamura","Kobayashi","Kato","Wang","Li","Zhang","Liu","Chen","Yang","Huang","Zhao","Wu","Zhou","Kim","Lee","Park","Choi","Jung","Kang","Cho","Yoon","Jang","Lim"]},
    "south_asian": {
        "M": ["Aarav","Vihaan","Arjun","Rohan","Kabir","Ishaan","Aditya","Rahul","Karan","Vikram","Arash","Reza","Ali","Omid","Timur","Aziz","Bekzod","Farhad","Sanjay","Amit"],
        "F": ["Ananya","Diya","Priya","Neha","Sneha","Pooja","Kavya","Meera","Riya","Isha","Sara","Mina","Dilnoza","Nilufar","Zarina","Shirin","Anjali","Divya","Lakshmi","Sunita"],
        "L": ["Sharma","Verma","Patel","Singh","Kumar","Gupta","Reddy","Nair","Iyer","Mehta","Khan","Rezaei","Hosseini","Ahmadi","Karimov","Rashidov","Aliyev","Nazarov","Mirza","Chopra","Joshi","Rao","Das","Bose","Kapoor","Tehrani","Moradi","Yusupov","Abdullaev","Seitkali"]},
    "african": {
        "M": ["Kwame","Kofi","Tunde","Chidi","Emeka","Sipho","Thabo","Ade","Femi","Kiptoo","Kipchoge","Haile","Mulugeta","Yared","Abebe","Samuel","Joseph","Moses","Brian","Peter"],
        "F": ["Abena","Zainab","Fatima","Ngozi","Chioma","Thandi","Lindiwe","Aisha","Nia","Zuri","Wanjiru","Jelimo","Tigist","Selam","Mercy","Grace","Faith","Mary","Ruth","Esther"],
        "L": ["Okafor","Adeyemi","Mensah","Boateng","Nkosi","Dlamini","Mokoena","Ndlovu","Osei","Owusu","Kamau","Otieno","Kiplagat","Chebet","Tadesse","Bekele","Tesfaye","Gebremariam","Mwangi","Ochieng","Hassan","Ibrahim","Mahmoud","Salah","Amrani","Benali","Mansour","Farouk","Nasser","Okello"]},
    "latam_caribbean": {
        "M": ["Juan","Miguel","Santiago","Sebastian","Mateo","Thiago","Rafael","Bruno","Davi","Gustavo","Usain","Asafa","Yohan","Omar","Andre","Marlon","Dwayne","Kemar","Carlos","Luis"],
        "F": ["Valentina","Camila","Isabella","Mariana","Fernanda","Beatriz","Larissa","Yasmin","Shelly","Shericka","Tamika","Simone","Daniela","Gabriela","Juliana","Leticia","Rosa","Alejandra","Paola","Marisol"],
        "L": ["Gonzalez","Rodriguez","Hernandez","Perez","Ramos","Mendoza","Rojas","Vega","Cruz","Morales","Souza","Lima","Carvalho","Almeida","Barbosa","Ribeiro","Nunes","Fraser","Bolt","Powell","Campbell","Blake","Thompson","Henry","Reid","Grant","Palmer","Francis","Wilson","Castro"]},
    "middle_east": {
        "M": ["Mohammed","Ahmed","Omar","Youssef","Hamza","Khalid","Faisal","Sultan","Tariq","Bilal","Emre","Mert","Burak","Kerem","Cem","Murat","Hakan","Yusuf","Ibrahim","Ali"],
        "F": ["Layla","Noor","Salma","Maryam","Huda","Rania","Dina","Yasmin","Amal","Leila","Elif","Zeynep","Ayse","Deniz","Selin","Merve","Esra","Fatma","Hatice","Buse"],
        "L": ["Al-Farsi","Al-Harbi","Al-Qahtani","Al-Otaibi","Haddad","Khoury","Nasser","Saleh","Yilmaz","Kaya","Demir","Sahin","Celik","Yildiz","Ozturk","Arslan","Dogan","Aydin","Kilic","Polat","Al-Zahrani","Al-Ghamdi","Mansour","Aziz","Farah","Suleiman","Bakr","Darwish","Nabulsi","Tamimi"]},
    "southeast_asian_pacific": {
        "M": ["Minh","Anh","Duc","Hung","Budi","Putra","Agus","Somchai","Chai","Aroon","Sione","Tevita","Manu","Semi","Batbayar","Enkhbat","Bold","Ganbold","Altan","Temuulen"],
        "F": ["Linh","Hoa","Lan","Mai","Dewi","Siti","Ayu","Nattaya","Pim","Malee","Ana","Litia","Mere","Sela","Narantsetseg","Tuya","Oyun","Solongo","Bolormaa","Khulan"],
        "L": ["Nguyen","Tran","Le","Pham","Hoang","Wijaya","Santoso","Saputra","Pratama","Susanto","Suksawat","Chaiyaporn","Wongsakul","Srisai","Tuilagi","Vaka","Taufa","Lolo","Ulufonua","Batbayar","Erdene","Munkh","Dorj","Ganbaatar","Tsend","Baatar","Ochir","Namjil","Sukh","Altangerel"]},
}
COUNTRY_TO_POOL = {}
def _assign(pool, countries):
    for c in countries: COUNTRY_TO_POOL[c] = pool
_assign("anglo", ["United States","Great Britain","Australia","Canada","New Zealand","Ireland"])
_assign("germanic", ["Germany","Netherlands","Sweden","Norway","Denmark","Switzerland","Belgium"])
_assign("romance", ["France","Italy","Spain","Portugal","Argentina","Mexico","Chile","Colombia","Costa Rica"])
_assign("slavic", ["Russia","Poland","Ukraine","Czechia","Croatia","Serbia","Slovenia","Lithuania","Hungary"])
_assign("east_asian", ["Japan","China","South Korea"])
_assign("south_asian", ["India","Iran","Uzbekistan","Kazakhstan"])
_assign("african", ["Kenya","Ethiopia","Nigeria","South Africa","Egypt","Morocco","Uganda"])
_assign("latam_caribbean", ["Brazil","Jamaica","Cuba"])
_assign("middle_east", ["Turkey","Saudi Arabia","Jordan"])
_assign("southeast_asian_pacific", ["Thailand","Vietnam","Indonesia","Fiji","Mongolia"])
for c in COUNTRY_NAMES:
    COUNTRY_TO_POOL.setdefault(c, "anglo")

# ---------------------------------------------------------------- athletes
def sample_names(countries, genders):
    """genders: array of 'M'/'F' or None (then coin flip)."""
    fulls = []
    for i, c in enumerate(countries):
        pool = NAME_POOLS[COUNTRY_TO_POOL[c]]
        g = genders[i] if genders is not None else ("M" if rng.random() < 0.6 else "F")
        first = pool[g][rng.integers(len(pool[g]))]
        last = pool["L"][rng.integers(len(pool["L"]))]
        fulls.append(f"{first} {last}")
    return fulls

print("Building athletes...")
countries = rng.choice(COUNTRY_NAMES, size=N_ATHLETES, p=C_WEIGHTS)
disciplines = rng.choice(D_NAMES, size=N_ATHLETES, p=D_WEIGHTS)

genders = np.empty(N_ATHLETES, dtype=object)
for d in D_NAMES:
    mask = disciplines == d
    m_p, w_p = DISCIPLINES[d][2]
    genders[mask] = rng.choice(["M", "F"], size=mask.sum(), p=[m_p, w_p])

names = sample_names(countries, genders)

# age by discipline (gymnastics/diving young, equestrian/shooting old)
age_mean = {d: 26 for d in D_NAMES}
age_mean.update({"Gymnastics": 20, "Diving": 22, "Swimming": 23, "Skateboarding": 20,
                 "Climbing": 23, "Surfing": 24, "Equestrian": 36, "Shooting": 34,
                 "Sailing": 31, "Archery": 28, "Triathlon": 28, "Table Tennis": 25})
ages = np.array([np.clip(rng.normal(age_mean[d], 4.5), 15, 52) for d in disciplines]).round().astype(int)

# height/weight by discipline & gender (cm / kg)
def body(d, g):
    tall = {"Basketball": 200, "Volleyball": 192, "Swimming": 185, "Rowing": 190,
            "Water Polo": 188, "Handball": 188, "Athletics": 178, "Football": 178,
            "Gymnastics": 165, "Diving": 168, "Weightlifting": 172, "Judo": 174,
            "Boxing": 174, "Wrestling": 174, "Equestrian": 172}
    base = tall.get(d, 175)
    h = rng.normal(base - (12 if g == "F" else 0), 6)
    bmi = rng.normal(22.5 + (1.5 if d in ("Weightlifting","Wrestling","Judo","Boxing") else 0), 1.8)
    w = bmi * (h / 100) ** 2
    return round(float(np.clip(h, 140, 225)), 0), round(float(np.clip(w, 38, 160)), 1)

hw = [body(d, g) for d, g in zip(disciplines, genders)]
heights = [x[0] for x in hw]
weights = [x[1] for x in hw]

# experience & debut
years_pro = np.clip(rng.gamma(3.0, 2.2, N_ATHLETES), 0, 25).round().astype(int)
debut = 2028 - years_pro
prior_games = np.clip(rng.poisson(1.1, N_ATHLETES), 0, 6)

# world ranking: lower = better, skewed by country tier
tier = np.array([C_TIER[c] for c in countries])
rank_noise = rng.lognormal(mean=4.2 - 2.0 * tier, sigma=0.8)
world_rank = np.clip(rank_noise.round().astype(int), 1, 2500)

athletes = pd.DataFrame({
    "AthleteID": [f"ATH{100000 + i}" for i in range(N_ATHLETES)],
    "PersonName": names,
    "Gender": genders,
    "Age": ages,
    "HeightCm": heights,
    "WeightKg": weights,
    "Country": countries,
    "Discipline": disciplines,
    "DebutYear": debut,
    "PriorGames": prior_games,
    "WorldRanking": world_rank,
})

# inject light realistic mess
miss_h = rng.random(N_ATHLETES) < 0.015
athletes.loc[miss_h, "HeightCm"] = np.nan
miss_w = rng.random(N_ATHLETES) < 0.02
athletes.loc[miss_w, "WeightKg"] = np.nan
miss_r = rng.random(N_ATHLETES) < 0.06
athletes.loc[miss_r, "WorldRanking"] = np.nan
athletes["WorldRanking"] = athletes["WorldRanking"].astype("Int64")

# ---------------------------------------------------------------- events, teams
print("Building events/teams...")
event_rows = []
for d in D_NAMES:
    _, is_team, (m_p, w_p), n_events, team_size = DISCIPLINES[d]
    for e in range(n_events):
        gender_cat = "Men" if e % 2 == 0 else "Women"
        if d in ("Tennis", "Badminton", "Table Tennis") and e == n_events - 1:
            gender_cat = "Mixed"
        event_rows.append((d, f"{d} Event {e+1:02d}", gender_cat, is_team, team_size))
events = pd.DataFrame(event_rows, columns=["Discipline", "Event", "Category", "IsTeam", "TeamSize"])

# Teams: for team disciplines, one team per country per event for enrolled countries
team_rows = []
for _, ev in events[events.IsTeam].iterrows():
    # enrol 8-24 countries per team event, weighted toward strong tiers
    n_c = int(rng.integers(8, 25))
    strong = rng.choice(COUNTRY_NAMES, size=n_c, replace=False,
                        p=(C_WEIGHTS * (0.5 + np.array([C_TIER[c] for c in COUNTRY_NAMES]))) /
                          (C_WEIGHTS * (0.5 + np.array([C_TIER[c] for c in COUNTRY_NAMES]))).sum())
    for c in strong:
        team_rows.append({
            "TeamName": f"{c} {ev.Category}" if ev.Category != "Mixed" else f"{c} Mixed",
            "Discipline": ev.Discipline,
            "Event": ev.Event,
            "Country": c,
            "SquadSize": int(np.clip(rng.normal(ev.TeamSize, 1.0), ev.TeamSize - 2, ev.TeamSize + 3)),
            "SeedRank": None,
        })
teams = pd.DataFrame(team_rows)
teams["SeedRank"] = teams.groupby("Event")["Country"].transform(
    lambda s: rng.permutation(len(s)) + 1)
# seed correlates with tier: reorder by tier + noise
def seed_by_strength(df):
    score = df["Country"].map(C_TIER) + rng.normal(0, 0.15, len(df))
    ranks = (-score).rank(method="first").astype(int)
    return ranks
teams["SeedRank"] = teams.groupby("Event", group_keys=False)["Country"].transform(
    lambda s: (-(s.map(C_TIER) + rng.normal(0, 0.15, len(s)))).rank(method="first").astype(int))
teams.insert(0, "TeamID", [f"TM{5000 + i}" for i in range(len(teams))])

# ---------------------------------------------------------------- coaches
print("Building coaches...")
# roughly 1 coach per 7 athletes, weighted by discipline & country
N_COACHES = N_ATHLETES // 7
c_countries = rng.choice(COUNTRY_NAMES, size=N_COACHES, p=C_WEIGHTS)
c_disc = rng.choice(D_NAMES, size=N_COACHES, p=D_WEIGHTS)
c_names = sample_names(c_countries, None)
coach_events = {
    d: [f"{d} Event {i+1:02d}" for i in range(DISCIPLINES[d][3])] for d in D_NAMES
}
c_event = [coach_events[d][rng.integers(len(coach_events[d]))] for d in c_disc]
c_role = rng.choice(["Head Coach", "Assistant Coach", "Technical Coach", "Fitness Coach"],
                    size=N_COACHES, p=[0.32, 0.35, 0.20, 0.13])
c_exp = np.clip(rng.gamma(4.0, 3.5, N_COACHES), 1, 45).round().astype(int)
coaches = pd.DataFrame({
    "CoachID": [f"CCH{200000 + i}" for i in range(N_COACHES)],
    "Name": c_names,
    "Country": c_countries,
    "Discipline": c_disc,
    "Event": c_event,
    "Role": c_role,
    "YearsExperience": c_exp,
})

# ---------------------------------------------------------------- entries by gender
print("Building EntriesGender...")
# One row per discipline with realistic men/women/total counts from athletes
ge = (athletes.groupby(["Discipline", "Gender"]).size().unstack(fill_value=0)
      .rename(columns={"F": "Female", "M": "Male"}).reset_index())
ge["Total"] = ge["Female"] + ge["Male"]
ge["FemalePct"] = (ge["Female"] / ge["Total"] * 100).round(1)
ge["MalePct"] = (ge["Male"] / ge["Total"] * 100).round(1)
ge = ge[["Discipline", "Female", "Male", "Total", "FemalePct", "MalePct"]]

# ---------------------------------------------------------------- medals
print("Simulating medal outcomes...")
# For each event: draw podium from countries weighted by tier and how many
# athletes/teams that country actually fielded in that discipline.
country_disc_counts = athletes.groupby(["Country", "Discipline"]).size()
medal_rows = []
for _, ev in events.iterrows():
    d = ev.Discipline
    if ev.IsTeam:
        entrants = teams[teams.Event == ev.Event]["Country"].tolist()
    else:
        # countries with a meaningful presence in this discipline
        entrants = [c for c in COUNTRY_NAMES if country_disc_counts.get((c, d), 0) >= 12]
    if len(entrants) < 3:
        entrants = COUNTRY_NAMES[:12]
    strength = np.array([
        (C_TIER[c] ** 4.5) * (country_disc_counts.get((c, d), 1) ** 0.55) for c in entrants
    ])
    strength = strength / strength.sum()
    podium = rng.choice(entrants, size=3, replace=False, p=strength)
    # occasional shared bronze (judo, boxing, wrestling, taekwondo) -> 2 bronzes
    two_bronze = d in ("Judo", "Boxing", "Wrestling", "Taekwondo")
    medals = [("Gold", podium[0]), ("Silver", podium[1]), ("Bronze", podium[2])]
    if two_bronze:
        extra = [c for c in entrants if c not in podium]
        if extra:
            ex_strength = np.array([C_TIER[c] for c in extra]); ex_strength = ex_strength / ex_strength.sum()
            medals.append(("Bronze", rng.choice(extra, p=ex_strength)))
    for m, c in medals:
        medal_rows.append({"Discipline": d, "Event": ev.Event, "Category": ev.Category,
                           "Medal": m, "Country": c})
medal_events = pd.DataFrame(medal_rows)

# add athlete of record for individual events (pick from that country's athletes in discipline)
by_cd = athletes.groupby(["Country", "Discipline"])["AthleteID"].apply(list).to_dict()
records = []
for r in medal_events.itertuples():
    ev_cat = r.Category
    if ev_cat == "Men":
        cand_g = "M"
    elif ev_cat == "Women":
        cand_g = "F"
    else:
        cand_g = None
    pool = by_cd.get((r.Country, r.Discipline), [])
    if cand_g is not None:
        pool = [a for a in pool if athletes.loc[int(a[3:]) - 100000, "Gender"] == cand_g]
    is_team = DISCIPLINES[r.Discipline][1]
    aid = pool[rng.integers(len(pool))] if (pool and not is_team) else None
    records.append(aid)
medal_events["AthleteID"] = records
medal_events.insert(0, "MedalID", [f"MED{9000 + i}" for i in range(len(medal_events))])
medals = medal_events

# medal tally, common in the tutorial (Rank, Team/NOC, Gold, Silver, Bronze, Total)
tally = (medals.pivot_table(index="Country", columns="Medal", values="MedalID",
                            aggfunc="count", fill_value=0)
         .reindex(columns=["Gold", "Silver", "Bronze"], fill_value=0))
tally["Total"] = tally.sum(axis=1)
tally = tally.sort_values(["Gold", "Silver", "Bronze"], ascending=False).reset_index()
tally.insert(0, "Rank", range(1, len(tally) + 1))
medals_tally = tally

# ---------------------------------------------------------------- save
print("Writing files...")
athletes.to_csv(OUT / "Athletes.csv", index=False)
coaches.to_csv(OUT / "Coaches.csv", index=False)
ge.to_csv(OUT / "EntriesGender.csv", index=False)
medals.to_csv(OUT / "Medals.csv", index=False)
teams.to_csv(OUT / "Teams.csv", index=False)
medals_tally.to_csv(OUT / "MedalsTally.csv", index=False)

print("\nDone.")
for name, df in [("Athletes", athletes), ("Coaches", coaches), ("EntriesGender", ge),
                 ("Medals", medals), ("Teams", teams), ("MedalsTally", medals_tally)]:
    print(f"  {name:14s} {len(df):>8,} rows  x {df.shape[1]} cols")
