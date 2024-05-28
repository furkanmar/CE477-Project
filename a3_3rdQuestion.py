import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'GameSales_dropped.csv'  # Verisetinin dosya yolu
data = pd.read_csv(file_path)


basket = data.groupby(['Publisher', 'Genre']).size().unstack().reset_index().fillna(0).set_index('Publisher')


def encode_units(x):
    return 1 if x >= 1 else 0

basket_sets = basket.applymap(encode_units).astype(bool)

# Apply the Apriori algorithm
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

# Extract association rules using support, confidence, and lift metrics
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2.0)

# Filter for interesting rules using lift and confidence
interesting_rules = rules[(rules['lift'] > 2.5) & (rules['confidence'] > 0.7)].copy()

# Convert 'lift' to float for nlargest method
interesting_rules['lift'] = interesting_rules['lift'].astype(float)

print("Interesting Association Rules:")
print(interesting_rules)

if not interesting_rules.empty:
    top_10_rules = interesting_rules.nlargest(10, 'lift')

    plt.figure(figsize=(12, 8))
    sns.barplot(x='lift', y=[f"{list(rule)[0]} -> {list(rule)[1]}" for rule in zip(top_10_rules['antecedents'], top_10_rules['consequents'])], data=top_10_rules, orient='h')
    plt.title('Top 10 Association Rules by Lift')
    plt.xlabel('Lift')
    plt.ylabel('Association Rule')
    plt.savefig('top_10_association_rules_by_lift.pdf')
    plt.show()
else:
    print("No interesting rules found.")