-- Employee Attrition Analytics Queries

-- 1. Attrition rate by department
SELECT 
    Department,
    COUNT(*) AS Total_Employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Attrition_Count,
    ROUND((SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS Attrition_Rate_Percentage
FROM EmployeeAttrition
GROUP BY Department
ORDER BY Attrition_Rate_Percentage DESC;

-- 2. Average salary by role
SELECT 
    JobRole,
    COUNT(*) AS Total_Employees,
    ROUND(AVG(MonthlyIncome), 2) AS Average_Monthly_Income
FROM EmployeeAttrition
GROUP BY JobRole
ORDER BY Average_Monthly_Income DESC;

-- 3. Overtime employee analysis
SELECT 
    OverTime,
    COUNT(*) AS Employee_Count,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Attrition_Count,
    ROUND((SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS Attrition_Rate
FROM EmployeeAttrition
GROUP BY OverTime;

-- 4. Gender attrition analysis
SELECT 
    Gender,
    COUNT(*) AS Total_Employees,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Attrition_Count,
    ROUND((SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) * 100.0) / COUNT(*), 2) AS Attrition_Rate
FROM EmployeeAttrition
GROUP BY Gender;

-- 5. Top departments with highest attrition and average satisfaction
SELECT 
    Department,
    SUM(CASE WHEN Attrition = 'Yes' THEN 1 ELSE 0 END) AS Total_Attrition,
    ROUND(AVG(JobSatisfaction), 2) AS Average_Job_Satisfaction
FROM EmployeeAttrition
GROUP BY Department
ORDER BY Total_Attrition DESC;
