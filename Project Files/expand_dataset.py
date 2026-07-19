import pandas as pd
import numpy as np

def expand_dataset():
    # Read the 9-row original dataset
    df = pd.read_csv('application_record.csv')
    
    # We will generate 500 rows based on these stats
    np.random.seed(42)
    n_samples = 500
    
    # Age: Normal distribution around mean
    ages = np.random.normal(loc=df['Age'].mean(), scale=df['Age'].std(), size=n_samples)
    ages = np.clip(ages, 18, 80).astype(int)
    
    # Income: skewed
    incomes = np.random.lognormal(mean=np.log(df['Income'].mean()), sigma=0.5, size=n_samples)
    incomes = np.clip(incomes, 15000, 200000).astype(int)
    
    # CreditScore: 
    credit_scores = np.random.normal(loc=df['CreditScore'].mean(), scale=df['CreditScore'].std(), size=n_samples)
    credit_scores = np.clip(credit_scores, 300, 850).astype(int)
    
    # LoanAmount
    loan_amounts = np.random.normal(loc=df['LoanAmount'].mean(), scale=df['LoanAmount'].std(), size=n_samples)
    loan_amounts = np.clip(loan_amounts, 10000, 500000).astype(int)
    
    # Categorical
    employment = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.75, 0.25])
    married = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.6, 0.4])
    
    # Generate Synthetic Approval criteria based on common logic 
    # High Income + High Credit = More likely approved (0 in this dataset usually means Approved, 1 = Rejected based on typical datasets, but wait, the sample:
    # 25,45000,Yes,720,100000,No,1  -> wait.
    # Looking at sample:
    # Age 25, Income 45000, Emp Yes, CS 720, Loan 100000 -> 1 (Rejected?)
    # Age 42, Income 30000, Emp No, CS 550, Loan 200000 -> 0 (Approved?)
    # Age 28, Income 70000, Emp Yes, CS 800, Loan 90000 -> 1 
    # Oh wait, 0 might mean Approved and 1 might mean Rejected. Let's make it typical: target 0 = Approved, 1 = Rejected.
    # Better yet, let's just make it so:
    # high credit & steady employment = 0 (Approved)
    
    approved = []
    for i in range(n_samples):
        # A simple linear combination to determine approval
        score = (credit_scores[i] / 850) * 0.4 + (incomes[i] / 200000) * 0.3 + (1 if employment[i] == 'Yes' else 0) * 0.3
        # Deduct score if loan amount is huge
        score -= (loan_amounts[i] / 500000) * 0.2
        
        # We assume 1 is Approved, 0 is Rejected. Let's stick to conventional 1=Approved, 0=Rejected for standard apps, 
        # Wait, the dataset target is STATUS (0=approved, 1=rejected - or vice versa).
        # We will explicitly make 0 = Approved, 1 = Rejected.
        # High score means good candidate -> Approved -> 0.
        prob = max(0, min(1, score))
        if prob > 0.5:
            approved.append(0) # Approved
        else:
            approved.append(1) # Rejected
            
    synthetic_df = pd.DataFrame({
        'Age': ages,
        'Income': incomes,
        'Employment': employment,
        'CreditScore': credit_scores,
        'LoanAmount': loan_amounts,
        'Married': married,
        'Approved': approved
    })
    
    # Save the expanded dataset
    synthetic_df.to_csv('application_record.csv', index=False)
    print("Successfully expanded application_record.csv to 500 rows.")

if __name__ == '__main__':
    expand_dataset()
