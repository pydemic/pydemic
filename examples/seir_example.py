from pydemic.all import *


# Initialize model
m = SEAIR(region="BR", disease=covid19, R0=1.5)
m.set_ic(cases=20000)
m.run(200)
cm = m.clinical.overflow_model()

# Compute empirical quantities
attack = cm["R:final"] / cm.population
attack_c = cm["cases:final"] / cm.population

print(f"Attack          : {pc(attack)}")
print(f"Attack (c)      : {pc(attack_c)}")
print(f"Qs              : {pc(cm.Qs)}")
print(f"CFR             : {pc(cm.CFR)}")
print(f"CFR (empirical) : {pc(cm.empirical_CFR)}")
print(f"IFR             : {pc(cm.IFR)}")
print(f"IFR             : {pc(cm.infection_fatality_ratio)}")
print(f"IFR (empirical) : {pc(cm.empirical_IFR)}")


# Show plots
plt.subplot(221)
m.plot(log=True)

plt.subplot(222)
cm.plot(["death_rate", "severe"], logy=True)

plt.subplot(223)
cm.plot(["infectious", "cases", "infected", "severe_cases", "critical_cases"], logy=True)

plt.subplot(224)
df = cm[["empirical_CFR", "empirical_IFR"]] * 100
df.plot(ax=plt.gca(), grid=True)

plt.show()
