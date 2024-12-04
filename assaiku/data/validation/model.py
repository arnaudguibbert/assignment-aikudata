import pandera as pa

from assaiku.data.constants.categories import *


class DataValidator(pa.DataFrameModel):
    age: pa.Int = pa.Field(
        ge=0, description="Age of the person, age 0 is possible"
    )
    class_of_worker: pa.Category = pa.Field(
        isin=CLASS_OF_WORKER, description="Class of worker"
    )
    detailed_industry_recode: pa.Category = pa.Field(
        isin=DETAILED_INDUSTRY_RECODE, description="Industry code"
    )
    detailed_occupation_recode: pa.Category = pa.Field(
        isin=DETAILED_OCCUPATION_RECODE, description="occupation recode"
    )
    education: pa.Category = pa.Field(
        isin=EDUCATION, description="Education level"
    )
    wage_per_hour: pa.Int = pa.Field(
        ge=0, description="How much the person is paid per hour"
    )
    enroll_in_edu_inst_last_wk: pa.Category = pa.Field(
        isin=ENROLL_IN_EDU_INST_LAST_WK
    )
    marital_stat: pa.Category = pa.Field(
        isin=MARITAL_STAT, description="Marital status"
    )
    major_industry_code: pa.Category = pa.Field(
        isin=MAJOR_INDUSTRY_CODE, description="Major industry code"
    )
    major_occupation_code: pa.Category = pa.Field(
        isin=MAJOR_OCCUPATION_CODE, description="Major occupation code"
    )
    race: pa.Category = pa.Field(isin=RACE, description="person race")
    hispanic_origin: pa.Category = pa.Field(
        isin=HISPANIC_ORIGIN,
        description="If hispanic, what's the person's origin, otherwise N/A",
    )
    sex: pa.Category = pa.Field(isin=SEX, description="Sex male or female")
    member_of_a_labor_union: pa.Category = pa.Field(
        isin=MEMBER_OF_A_LABOR_UNION,
        description="Was the person a member of labor union in its life ?",
    )
    reason_for_unemployment: pa.Category = pa.Field(
        isin=REASON_FOR_UNEMPLOYMENT,
        description="What was the reason for unemployment if the person was unemployed ?",
    )
    full_or_part_time_employment_stat: pa.Category = pa.Field(
        isin=FULL_OR_PART_TIME_EMPLOYMENT_STAT,
        description="Is the person full time or part time",
    )
    capital_gains: pa.Int = pa.Field(
        ge=0.0,
        description="How much money did the person earned from capital gain (last year)",
    )
    capital_losses: pa.Int = pa.Field(
        ge=0,
        description="How much money did the person lossed from capital gain (last year)",
    )
    dividends_from_stocks: pa.Int = pa.Field(
        ge=0.0, description="MOney from dividend from stocks"
    )
    tax_filer_stat: pa.Category = pa.Field(
        isin=TAX_FILER_STAT, description="Tax filer status"
    )
    region_of_previous_residence: pa.Category = pa.Field(
        isin=REGION_OF_PREVIOUS_RESIDENCE,
        description="Region of previous residence, if nothing -> previous in Universe",
    )
    state_of_previous_residence: pa.Category = pa.Field(
        isin=STATE_OF_PREVIOUS_RESIDENCE,
        description="State of previous residence",
    )
    detailed_household_and_family_stat: pa.Category = pa.Field(
        isin=DETAILED_HOUSEHOLD_AND_FAMILY_STAT, description="Family settings"
    )
    detailed_household_summary_in_household: pa.Category = pa.Field(
        isin=DETAILED_HOUSEHOLD_SUMMARY_IN_HOUSEHOLD,
        description="Migration code",
    )
    instance_weight: pa.Float = pa.Field(
        gt=0.0,
        description="Number of people in the population with such attributes, should not be used to prediction",
    )
    migration_code_change_in_msa: pa.Category = pa.Field(
        isin=MIGRATION_CODE_CHANGE_IN_MSA,
        description="Migration code change in msa",
    )
    migration_code_change_in_reg: pa.Category = pa.Field(
        isin=MIGRATION_CODE_CHANGE_IN_REG,
        description="Migration code change in reg",
    )
    migration_code_move_within_reg: pa.Category = pa.Field(
        isin=MIGRATION_CODE_MOVE_WITHIN_REG,
        description="Migration code move within reg",
    )
    live_in_this_house_1_year_ago: pa.Category = pa.Field(
        isin=LIVE_IN_THIS_HOUSE_1_YEAR_AGO,
        description="Did the person live in this house 1 year ago ?",
    )
    migration_prev_res_in_sunbelt: pa.Category = pa.Field(
        isin=MIGRATION_PREV_RES_IN_SUNBELT,
        description="Did you previous residency was in sunbelt ?",
    )
    num_persons_worked_for_employer: pa.Int = pa.Field(
        ge=0, description="Number of employers you worked for"
    )
    family_members_under_18: pa.Category = pa.Field(
        isin=FAMILY_MEMBERS_UNDER_18, description="Parents present in the house"
    )
    country_of_birth_father: pa.Category = pa.Field(
        isin=COUNTRY_OF_BIRTH_FATHER, description="Country of birth father"
    )
    country_of_birth_mother: pa.Category = pa.Field(
        isin=COUNTRY_OF_BIRTH_MOTHER, description="Country of birth mother"
    )
    country_of_birth_self: pa.Category = pa.Field(
        isin=COUNTRY_OF_BIRTH_SELF, description="Country of birth of the person"
    )
    citizenship: pa.Category = pa.Field(
        isin=CITIZENSHIP, description="Citizenship of the person"
    )
    own_business_or_self_employed: pa.Category = pa.Field(
        isin=OWN_BUSINESS_OR_SELF_EMPLOYED,
        description="Does the person have its own business, 0 -> niu, 1 -> yes, 2 -> no",
    )
    fill_inc_questionnaire_for_veterans_admin: pa.Category = pa.Field(
        isin=FILL_INC_QUESTIONNAIRE_FOR_VETERANS_ADMIN,
        description="Did person require to fill a veteran questionnaire ?",
    )
    veterans_benefits: pa.Category = pa.Field(
        isin=VETERANS_BENEFITS,
        description="Did the person receive veteran payments ? 0 niu, 1 yes, 2 no",
    )
    weeks_worked_in_year: pa.Int = pa.Field(
        ge=0, description="Number of weeks the person worked in a day"
    )
    year: pa.Category = pa.Field(isin=YEAR, description="Year")
    income: pa.Category = pa.Field(
        isin=INCOME,
        description="Does the person owns more or less than 50000k a year ?",
    )
