# sql_generator.py
from vanna_provider import generate_sql_from_prompt
from schema_service import schema_service

BASE_SQL_RULES = """
You are generating T-SQL for a Microsoft SQL Server star-schema data warehouse
(e.g., AdventureWorksDW).

GLOBAL RULES:

1. TIME FILTERS
   - ALWAYS join fact tables to DimDate using:
       ON <fact>.OrderDateKey = DimDate.DateKey
   - ALWAYS filter years using:
       DimDate.CalendarYear = <year>
       OR DimDate.CalendarYear BETWEEN X AND Y
   - NEVER compare OrderDateKey to string dates (e.g., '2003-01-01').

2. GROUPING RULES
   - Only group by CalendarYear when explicitly requested (e.g. “by year”).
   - If grouping by product, include:
       Product.ProductKey,
       Product.EnglishProductName
   - If grouping by category or subcategory, include only what is needed.
   - DO NOT add extra grouping columns not requested.

3. DIMENSION RULES
   - Always join facts → dimensions using the correct FK keys:
       FactInternetSales.ProductKey = DimProduct.ProductKey
       DimProduct.ProductSubcategoryKey → DimProductSubcategory
       DimProductSubcategory.ProductCategoryKey → DimProductCategory

4. OUTPUT RULES
   - Only output columns explicitly required or needed for the grouping.
   - Use real schema column names exactly as they appear.
   - NEVER invent columns or tables.

5. GENERAL T-SQL RULES
   - Do NOT use: LIMIT, OFFSET, QUALIFY, GROUPING SETS, CUBE.
   - CTEs are allowed.
"""

def generate_sql(question: str) -> str:
    """
    High-level NL → SQL conversion with strong rules + schema context.
    All providers (LocalVanna, OpenAI, etc.) pass through here.
    """
    prompt = f"""
{BASE_SQL_RULES}

QUESTION:
{question}

SCHEMA:
{schema_service.schema_text}

INSTRUCTIONS:
- Generate ONE valid T-SQL SELECT query.
- Use only tables + columns from the schema.
- Use DimDate for all date/year filters.
- Do NOT invent tables or columns.
- No explanations or comments.

Return ONLY the SQL.
"""

    return generate_sql_from_prompt(prompt)
