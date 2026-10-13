* SPSS Syntax - Student Performance Prediction.
* STEP 1: Edit the path below to where student_data.csv is saved, then Run > All.

GET DATA /TYPE=TXT
  /FILE='C:\Project\student_data.csv'
  /DELIMITERS=","
  /QUALIFIER='"'
  /ARRANGEMENT=DELIMITED
  /FIRSTCASE=2
  /VARIABLES=
    study_hours F5.1
    attendance F5.1
    previous_score F5.1
    sleep_hours F4.1
    absences F3.0
    internet_access A3
    parental_education A12
    final_score F5.1.
EXECUTE.

* Recode text variables into numbers for modelling.
RECODE internet_access ('Yes'=1) ('No'=0) INTO internet_num.
RECODE parental_education ('School'=1) ('Graduate'=2) ('Postgraduate'=3) INTO parent_edu_num.
EXECUTE.

* 1. Descriptive statistics and missing values check.
DESCRIPTIVES VARIABLES=study_hours attendance previous_score sleep_hours absences final_score
  /STATISTICS=MEAN STDDEV MIN MAX.
FREQUENCIES VARIABLES=internet_access parental_education.

* 2. Graphs.
GRAPH /HISTOGRAM(NORMAL)=final_score.
GRAPH /SCATTERPLOT(BIVAR)=study_hours WITH final_score.

* 3. Correlation.
CORRELATIONS /VARIABLES=final_score study_hours attendance previous_score sleep_hours absences
  /PRINT=TWOTAIL NOSIG.

* 4. Multiple Linear Regression (80:20 split is shown in the Python part).
REGRESSION
  /STATISTICS COEFF OUTS R ANOVA COLLIN TOL
  /DEPENDENT final_score
  /METHOD=ENTER study_hours attendance previous_score sleep_hours absences internet_num parent_edu_num
  /SCATTERPLOT=(*ZRESID ,*ZPRED)
  /RESIDUALS HISTOGRAM(ZRESID) NORMPROB(ZRESID)
  /SAVE PRED.

* 5. Group comparison: ANOVA of score by parental education.
ONEWAY final_score BY parental_education /STATISTICS DESCRIPTIVES /POSTHOC=TUKEY ALPHA(0.05).
