import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def create_eda_plots(csv_path, output_dir):
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for premium look
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (8, 6)
    
    # 1. Count Plot: Employment
    plt.figure()
    sns.countplot(data=df, x='Employment', palette='viridis')
    plt.title('Count of Employment Status')
    plt.savefig(os.path.join(output_dir, 'employment_count.png'), bbox_inches='tight', transparent=True)
    plt.close()
    
    # 2. Count Plot: Married
    plt.figure()
    sns.countplot(data=df, x='Married', palette='magma')
    plt.title('Count of Marital Status')
    plt.savefig(os.path.join(output_dir, 'married_count.png'), bbox_inches='tight', transparent=True)
    plt.close()
    
    # 3. Count Plot: Approved
    plt.figure()
    sns.countplot(data=df, x='Approved', palette='Set2')
    plt.title('Count of Approval Status')
    plt.savefig(os.path.join(output_dir, 'approved_count.png'), bbox_inches='tight', transparent=True)
    plt.close()
    
    # 4. Distribution Plot: Age
    plt.figure()
    sns.histplot(df['Age'], kde=True, color='skyblue', bins=20)
    plt.title('Age Distribution')
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), bbox_inches='tight', transparent=True)
    plt.close()
    
    # 5. Distribution Plot: Income
    plt.figure()
    sns.histplot(df['Income'], kde=True, color='lightgreen', bins=20)
    plt.title('Income Distribution')
    plt.savefig(os.path.join(output_dir, 'income_distribution.png'), bbox_inches='tight', transparent=True)
    plt.close()
    
    # 6. Distribution Plot: CreditScore
    plt.figure()
    sns.histplot(df['CreditScore'], kde=True, color='salmon', bins=20)
    plt.title('Credit Score Distribution')
    plt.savefig(os.path.join(output_dir, 'credit_score_distribution.png'), bbox_inches='tight', transparent=True)
    plt.close()
    
    # 7. Distribution Plot: LoanAmount
    plt.figure()
    sns.histplot(df['LoanAmount'], kde=True, color='purple', bins=20)
    plt.title('Loan Amount Distribution')
    plt.savefig(os.path.join(output_dir, 'loan_amount_distribution.png'), bbox_inches='tight', transparent=True)
    plt.close()
    
    print("EDA plots generated successfully in", output_dir)

if __name__ == "__main__":
    create_eda_plots('application_record.csv', 'static/images')
