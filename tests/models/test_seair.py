from pydemic.models import SEAIR
from pydemic.diseases import covid19


class TestSEAIR:
    def test_seair_initialization(self):
        # Probability of developing symptoms
        assert SEAIR(disease="disease").prob_symptoms == 1.0
        assert SEAIR(disease=covid19).prob_symptoms == covid19.prob_symptoms()
        assert SEAIR(prob_symptoms=0.5).prob_symptoms == 0.5
        assert SEAIR(Qs=0.5).prob_symptoms == 0.5
