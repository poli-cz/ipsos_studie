# load csv files clones_24.csv

import pandas as pd
import json

# df = pd.read_csv("clones_24.csv")
df = pd.read_csv("clones_full.csv")

# load column favourite_brands and load it into json

missing = 0

total_income = 0
income_missing = 0

sd = 0
# dict for storing political_party and count
parties = {}

average_energy = 0
energy_hassatribute = 0

# household statistics
household = {}
household_missing = 0


# compute average age from df["age"] and sd
# for age

energy_sd = 0


age = df["age"]
average_age = sum(age) / len(age)

sd_age = 0

for i in age:
    sd_age += (i - average_age) ** 2

sd_age = (sd_age / len(age)) ** 0.5

print(f"Average age: {average_age}")
print(f"SD for age: {sd_age}")

for i, index in enumerate(df["favorite_brands"]):

    energy = json.loads(df["energy"][i])
    brands = json.loads(df["favorite_brands"][i])
    household_data = json.loads(df["household"][i])
    economy = json.loads(df["economic_status"][i])

    try:
        average_energy += energy["avg_energy_usage_kWh"]
        energy_hassatribute += 1

        # compute sd for energy
        energy_sd += energy["avg_energy_usage_kWh"] ** 2

    except KeyError:
        pass

    # parse legal_form_of_use inside household
    try:
        legal_form = household_data["legal_form_of_use"]
        if legal_form in household:
            household[legal_form] += 1
        else:
            household[legal_form] = 1
    except KeyError:
        household_missing += 1

    # parse income
    try:
        total_income += economy["income"]
        # compute sd for income
        sd += economy["income"] ** 2

    except KeyError:
        income_missing += 1

    # check if brands is not empty

    try:
        party = brands[
            "Jaká_je_Vaše_nejoblíbenější_strana?hodnoťte_bez_ohledu_na_to,_zda_jí_volíte_či_nikoliv._o1"
        ]

        if party in parties:
            parties[party] += 1
        else:
            parties[party] = 1

    except KeyError:
        missing += 1
        print(df["age"][i])
        print(f"So far missing: {missing}")


# print percentage of missing values
print(
    f"Average energy consumption for CZ per person: {average_energy / energy_hassatribute} KWh"
)


# print average income
print(f"Average income: {total_income / (len(df) - income_missing)} CZK")

# compute SD for income
print(
    "SD for income: ",
    (sd / (len(df) - income_missing) - (total_income / (len(df) - income_missing)) ** 2)
    ** 0.5,
)
# compute sd for energy

sd_energy = (
    energy_sd / energy_hassatribute - (average_energy / energy_hassatribute) ** 2
) ** 0.5
print(f"SD for energy: {sd_energy}")

# print percentage statistics for household


# create bar chart with percentage of political parties

# as base number take df lenght minus missing values
# as percentage take number of occurences of party divided by base number
# and multiply by 100


total_people = len(df) - missing

for party in parties:
    parties[party] = parties[party] / total_people * 100

# create chart and sort it by values
import matplotlib.pyplot as plt

parties = dict(sorted(parties.items(), key=lambda item: item[1], reverse=True))

# show percentage under each bar
for party in parties:
    plt.bar(party, parties[party])
    plt.text(party, parties[party], f"{parties[party]:.2f}%", ha="center")

plt.xticks(rotation=90)
plt.ylabel("Percentage")
plt.title("Political parties")
plt.show()


# plot household statistics
total_households = len(df) - household_missing

for legal_form in household:
    household[legal_form] = household[legal_form] / total_households * 100

household = dict(sorted(household.items(), key=lambda item: item[1], reverse=True))

for legal_form in household:
    plt.bar(legal_form, household[legal_form])
    plt.text(
        legal_form, household[legal_form], f"{household[legal_form]:.2f}%", ha="center"
    )

plt.xticks(rotation=90)
plt.ylabel("Percentage")
plt.title("Legal forms of use")
plt.show()
