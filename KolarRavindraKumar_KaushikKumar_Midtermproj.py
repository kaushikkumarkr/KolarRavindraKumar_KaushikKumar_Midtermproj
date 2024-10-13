#Importing libraries
import pandas as pd
import itertools
from collections import defaultdict
import time
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

# Load dataset from CSV file
def load_data(file_path):
    df = pd.read_csv(file_path)
    transactions = df.apply(lambda x: set(x.dropna().astype(str)), axis=1).tolist()  # Ensure all items are strings
    return transactions

# Generate 1-itemsets for Brute Force
def generate_1_itemsets(transactions):
    itemset = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            itemset[item] += 1
    return {frozenset([item]): count for item, count in itemset.items()}

# Generate k-itemsets from (k-1)-itemsets
def generate_k_itemsets(prev_itemsets, k):
    new_itemsets = set()
    prev_itemsets = list(prev_itemsets.keys())
    
    for i in range(len(prev_itemsets)):
        for j in range(i + 1, len(prev_itemsets)):
            candidate = prev_itemsets[i] | prev_itemsets[j]
            if len(candidate) == k:
                new_itemsets.add(candidate)
    
    return new_itemsets

# Count support for itemsets
def count_support(transactions, itemsets):
    itemset_count = defaultdict(int)
    
    for transaction in transactions:
        for itemset in itemsets:
            if itemset.issubset(transaction):
                itemset_count[itemset] += 1
                
    return itemset_count

# Generate frequent itemsets for Brute Force
def generate_frequent_itemsets(transactions, min_support):
    frequent_itemsets = {}
    total_transactions = len(transactions)
    k = 1

    # Generate 1-itemsets
    current_itemsets = generate_1_itemsets(transactions)

    while current_itemsets:
        # Filter itemsets based on min_support
        frequent_itemsets_k = {itemset: count for itemset, count in current_itemsets.items() if (count / total_transactions) * 100 >= min_support}
        if not frequent_itemsets_k:
            break
        
        # # Print the current k-itemsets and their counts
        # print(f"\n{k}-itemsets:")
        # for itemset, count in frequent_itemsets_k.items():
        #     support = (count / total_transactions) * 100
        #     print(f"Itemset: {set(itemset)}, Count: {count}, Support: {support:.2f}%")
        
        frequent_itemsets.update(frequent_itemsets_k)
        
        # Generate next k-itemsets
        k += 1
        current_candidates = generate_k_itemsets(frequent_itemsets_k, k)
        current_itemsets = count_support(transactions, current_candidates)
    
    return frequent_itemsets

# Generate association rules from frequent itemsets
def generate_association_rules(frequent_itemsets, min_confidence, total_transactions):
    rules = []
    
    for itemset, support_count in frequent_itemsets.items():
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent
                
                if consequent:
                    confidence = (support_count / frequent_itemsets[antecedent]) * 100
                    support = (support_count / total_transactions) * 100
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, support, confidence))
    
    return rules

# Encode transactions to one-hot format
def encode_transactions(transactions):
    items = sorted(set(item for transaction in transactions for item in transaction))
    encoded_df = pd.DataFrame(0, index=range(len(transactions)), columns=items)
    
    for idx, transaction in enumerate(transactions):
        for item in transaction:
            encoded_df.at[idx, item] = 1
    
    return encoded_df

import time


#EXTRACTING AND COMPARING RESULTS
# Function to transform rules into a consistent format for comparison (antecedents, consequents, support, confidence)
def format_rules(rule_list, brute_force=False):
    formatted_rules = set()
    if brute_force:
        # Brute force rules already come as tuples (antecedent, consequent, support, confidence)
        for antecedent, consequent, support, confidence in rule_list:
            formatted_rules.add((frozenset(antecedent), frozenset(consequent), round(support, 2), round(confidence, 2)))
    else:
        # For Apriori and FP-Growth, rules are in DataFrame format with 'antecedents', 'consequents', 'support', 'confidence'
        for _, row in rule_list.iterrows():
            formatted_rules.add((frozenset(row['antecedents']), frozenset(row['consequents']), round(row['support'] * 100, 2), round(row['confidence'] * 100, 2)))
    return formatted_rules

# Function to compare the association rules across algorithms
def compare_algorithm_rules(brute_force_rules, apriori_rules, fp_growth_rules):
    # Format the rules from each approach
    brute_force_set = format_rules(brute_force_rules, brute_force=True)
    apriori_set = format_rules(apriori_rules)
    fp_growth_set = format_rules(fp_growth_rules)

    # Check if all sets of rules are identical
    identical_rules = brute_force_set == apriori_set == fp_growth_set

    if identical_rules:
        print("\nThe algorithms produced exactly the same association rules (matching antecedents, consequents, support, and confidence).")
    else:
        print("\nThe algorithms generated differing association rules.")

        # Display specific differences for further analysis
        print("\nRules unique to Brute Force:")
        print(brute_force_set - apriori_set - fp_growth_set)
        
        print("\nRules unique to Apriori:")
        print(apriori_set - brute_force_set - fp_growth_set)
        
        print("\nRules unique to FP-Growth:")
        print(fp_growth_set - brute_force_set - apriori_set)



# MAIN FUNCTION
# Assume load_data, generate_frequent_itemsets, generate_association_rules, encode_transactions, apriori, association_rules, and fpgrowth are already defined functions

def main():
    # Dictionary of available datasets
    datasets = {
        1: 'data/Amazon.csv',  # Amazon
        2: 'data/Costco.csv',  # Costco
        3: 'data/Levis.csv',  # Levis
        4: 'data/Nike.csv',  # Nike
        5: 'data/Walmart.csv'   # Walmart
    }
    
    # Loop until valid input or exit
    while True:
        print("Please choose a dataset:")
        for key, name in zip(datasets.keys(), ['Amazon', 'Costco', 'Levis', 'Nike', 'Walmart']):
            print(f"{key}. {name}")
        print("6. Exit")  # Option to exit

        try:
            choice = int(input("Enter the number corresponding to the dataset: "))
            if choice == 6:
                print("Exiting program.")
                return  # Exit the program
            elif choice in datasets:
                file_path = datasets[choice]
                print(f"Loading data")
                transactions = load_data(file_path)
                break  # Valid dataset chosen, exit the loop
            else:
                print("Invalid choice. Please select a valid dataset.")
        except ValueError:
            print("Invalid input. Please enter a number corresponding to the dataset.")

    # Input support and confidence thresholds
    min_support = float(input("Enter minimum support threshold (0-100): "))
    print("Support = ",min_support,"%")
    min_confidence = float(input("Enter minimum confidence threshold (0-100): "))
    print("Confidence = ",min_confidence,"%")

    # Brute Force Approach
    start_time = time.time()
    frequent_itemsets_bf = generate_frequent_itemsets(transactions, min_support)
    total_transactions = len(transactions)
    association_rules_bf = generate_association_rules(frequent_itemsets_bf, min_confidence, total_transactions)
    brute_force_time = time.time() - start_time

    # Apriori Approach
    start_time = time.time()
    encoded_df = encode_transactions(transactions)
    frequent_itemsets_apriori = apriori(encoded_df, min_support=min_support / 100, use_colnames=True)
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence / 100)
    apriori_time = time.time() - start_time

    # FP-Growth Approach
    start_time = time.time()
    frequent_itemsets_fp = fpgrowth(encoded_df, min_support=min_support / 100, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=min_confidence / 100)
    fp_growth_time = time.time() - start_time

    # Display results
    print("\nBrute Force Results:")
    for antecedent, consequent, support, confidence in association_rules_bf:
        print(f"Rule: {set(antecedent)} -> {set(consequent)}, Support: {support:.2f}%, Confidence: {confidence:.2f}%")

    print("\nApriori Results:")
    print(rules_apriori[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    print("\nFP-Growth Results:")
    print(rules_fp[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    # Comparison of results (calling the comparison function)
    compare_algorithm_rules(association_rules_bf, rules_apriori, rules_fp)


    # Timing performance
    print(f"\nTiming Performance:")
    print(f"Brute Force Time: {brute_force_time:.4f} seconds")
    print(f"Apriori Time: {apriori_time:.4f} seconds")
    print(f"FP-Growth Time: {fp_growth_time:.4f} seconds")

    # Determine which algorithm is fastest
    fastest = min(("Brute Force", brute_force_time), ("Apriori", apriori_time), ("FP-Growth", fp_growth_time), key=lambda x: x[1])
    print(f"\nFastest Algorithm: {fastest[0]} with a time of {fastest[1]:.4f} seconds")


if __name__ == "__main__":
    main()
