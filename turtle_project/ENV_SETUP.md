# Environment Setup for Turtle Trading Project

## API Keys Configuration

To use the data acquisition features, you'll need to set up API keys in a `.env` file.

### Step 1: Create .env file
Create a file named `.env` in the turtle_project root directory:

```bash
# In turtle_project directory
touch .env
```

### Step 2: Add API Keys
Add the following content to your `.env` file:

```bash
# Alpha Vantage API Key (primary data source)
ALPHAVANTAGE_API_KEY=your_actual_api_key_here

# Optional: Other API keys for future use
# QUANDL_API_KEY=your_quandl_key_here
# FRED_API_KEY=your_fred_key_here

# Project configuration
PROJECT_NAME=turtle_trading
DATA_DIR=./data
```

### Step 3: Get Alpha Vantage API Key
1. Visit: https://www.alphavantage.co/support/#api-key
2. Sign up for a free account
3. Copy your API key
4. Replace `your_actual_api_key_here` in the `.env` file

### Fallback Option
If you don't have an API key, the code will automatically fall back to using `yfinance` (Yahoo Finance) for data acquisition. This works without any API key but may have some limitations.

### Security Notes
- ✅ The `.env` file is already in `.gitignore` and will NOT be committed to git
- ✅ Never share your API keys publicly
- ✅ The free Alpha Vantage tier allows 5 calls/minute, 500 calls/day

### Verification
You can verify your setup by running the project setup notebook:
```bash
jupyter notebook notebooks/00_project_setup.ipynb
```

The notebook will show "API_KEY present: True" if your .env file is configured correctly. 