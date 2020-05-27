from pytest import approx as _approx

import mundi
from pydemic.diseases import disease as get_disease
from pydemic.models import SIR


class TestResults:
    def test_results(self):
        approx = lambda x: _approx(x, rel=0.005)
        m = SIR(disease="covid-19", region="BR")
        m.set_ic(cases=1e6)
        m.run(60)

        # Data
        assert m.results["data.attack_rate"] == approx(0.919)
        assert m.results["data.cases"] == approx(194236068)
        assert m.results["data.infectious"] == approx(4045)
        assert m.results["data.recovered"] == approx(194632955)
        assert m.results["data.susceptible"] == approx(17118720)
        assert m.results["data"] == approx(
            {
                "attack_rate": 0.919,
                "cases": 194236068,
                "infectious": 4045,
                "recovered": 194632955,
                "susceptible": 17118720,
            }
        )

        # Parameters
        assert m.results["params.R0"] == m.R0
        assert m.results["params.gamma"] == m.gamma
        assert m.results["params.infectious_period"] == m.infectious_period
        assert m.results["params"] == approx({"R0": 2.74, "infectious_period": 3.47})

        # Dates
        dates = m.results["dates"]
        assert dates["start"] < dates["peak"] < dates["end"]

    def test_can_store_information_in_results(self):
        m = SIR(disease="covid-19", region="BR")
        m.run(10)

        expect = {"bar": 42, "spam": "eggs"}
        m.results["foo.bar"] = 42
        m.results["foo.spam"] = "eggs"
        assert m.results["foo"] == expect
        assert "foo" in m.results.keys()

        assert "foo" in m.results.to_dict()
        assert "foo.bar" in m.results.to_dict(flat=True)

        m.run(10)
        assert m.results.get("foo") is None


class TestInfo:
    def test_info(self):
        approx = lambda x: _approx(x, rel=0.005)
        disease = get_disease("covid-19")
        m = SIR(disease="covid-19", region="BR")
        m.set_ic(cases=1e6)
        m.run(60)

        # Disease
        infectious_period = disease.infectious_period(region="BR")
        assert m.info["disease.CFR"] == approx(disease.CFR(region="BR"))
        assert m.info["disease.IFR"] == approx(disease.IFR(region="BR"))
        assert m.info["disease.infectious_period"] == approx(infectious_period)
        assert set(m.info["disease"]) == {
            "R0",
            "case_fatality_ratio",
            "critical_delay",
            "critical_period",
            "death_delay",
            "hospital_fatality_ratio",
            "hospitalization_overflow_bias",
            "hospitalization_period",
            "hospitalization_table",
            "icu_fatality_ratio",
            "icu_period",
            "incubation_period",
            "infection_fatality_ratio",
            "infectious_period",
            "mortality_table",
            "prob_aggravate_to_icu",
            "prob_critical",
            "prob_severe",
            "prob_symptoms",
            "rho",
            "severe_delay",
            "severe_period",
            "symptom_delay",
        }

        # Region
        br = mundi.region("BR")
        keys = {
            "population",
            "age_distribution",
            "age_pyramid",
            "hospital_capacity",
            "icu_capacity",
        }

        assert m.info["region.population"] == br.population
        assert all(m.info["region.age_distribution"] == br.age_distribution)
        assert all(m.info["region.age_pyramid"] == br.age_pyramid)
        assert set(m.info["region"]) == keys
