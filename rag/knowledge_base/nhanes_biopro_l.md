# NHANES BIOPRO_L dataset notes

Source: https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BIOPRO_L.htm

BIOPRO_L is the Standard Biochemistry Profile public data file from the NHANES
August 2021 through August 2023 cycle. The documentation describes laboratory
methods, quality assurance, data processing, analytical notes, and variable
information for the public release.

NHANES notes that not every participant provided biospecimens or enough volume
for every test. Missingness therefore needs to be assessed for the analytic
purpose rather than assigned an assumed cause.

The file includes a phlebotomy weight, WTPH2YR. The documentation states that
it is relevant to population analyses using blood analytes and to nonresponse
considerations. LabInsight does not use this survey weight in its individual
statistical anomaly score.

For analytes with results below a lower limit of detection, NHANES documents an
imputed fill value and a related comment variable. LabInsight excludes comment
codes from the 18-feature anomaly panel.
