from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def build_model():
    model = BayesianNetwork([("S", "O"), ("S", "M"), ("S", "L"), ("L", "M")])

    # S: spam sau nu
    # O: oferta sau nu
    # M: lungimea este mare sau nu
    # L: contine link-uri

    cpd_s = TabularCPD(variable="S", variable_card=2, values=[[0.4], [0.6]])

    cpd_o = TabularCPD(variable="O", variable_card=2,
                       values=[[0.7, 0.4],
                               [0.3, 0.6]],
                       evidence=["S"], evidence_card=[2])

    cpd_l = TabularCPD(variable="L", variable_card=2,
                       values=[[0.8, 0.1],
                               [0.2, 0.9]],
                       evidence=["S"], evidence_card=[2])

    cpd_m = TabularCPD(variable="M", variable_card=2,
                       values=[[0.5, 0.4, 0.7, 0.1],
                               [0.5, 0.6, 0.3, 0.9]],
                       evidence=["S", "L"], evidence_card=[2, 2])

    model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)
    model.check_model()
    return model

def check_dependencies(model):
    print(model.local_independencies(["S", "O", "M", "L"]))

def do_classification(model):

    infer = VariableElimination(model)

    # for loop prin fiecare 'atribut' posibil
    for o_value in [0, 1]:  # O: oferta or not
        for l_value in [0, 1]:  # L: contine link-uri sau nu
            for m_value in [0, 1]:  # M: mare sau nu
                # inference and classify the email
                result = infer.query(variables=['S'], evidence={'O': o_value, 'L': l_value, 'M': m_value})

                # get probabilities
                prob_spam = result.values[1]  # P(S=1 | evidence)
                prob_non_spam = result.values[0]  # P(S=0 | evidence)

                # determine classification
                classification = 'Spam' if prob_spam > prob_non_spam else 'Non-Spam'

                # the results
                print(f"O={o_value}, L={l_value}, M={m_value} -> {classification}")
                print(f"  P(Spam) = {prob_spam}, P(Non-Spam) = {prob_non_spam}")
                print("-" * 40)


if __name__ == '__main__':
    email_model = build_model()

    check_dependencies(email_model)

    do_classification(email_model)
