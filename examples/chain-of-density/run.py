import instructor
import openai
from pydantic import BaseModel, Field

from pprint import pprint
from pydantic import BaseModel, Field
from typing import List, Dict


class Summary(BaseModel):
    """Represents a summary entry in the list.

    Guidelines:
        - The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific,
          containing little information beyond the entities marked as missing. Use overly verbose
          language and fillers (e.g., "this article discusses") to reach ~80 words.
        - Make every word count: rewrite the previous summary to improve flow and make space for
          additional entities.
        - Make space with fusion, compression, and removal of uninformative phrases like "the article discusses."
        - The summaries should become highly dense and concise yet self-contained, i.e., easily understood
          without the article.
        - Missing entities can appear anywhere in the new summary.
        - Never drop entities from the previous summary. If space cannot be made, add fewer new entities.
    """

    index: int = Field(..., description="Index of the summary in the chain.")
    denser_summary: str = Field(..., description="Concise yet self-contained summary.")
    included_entities: List[str] = Field(
        ..., description="Correct list of Entities found in the summary."
    )
    missing_entities: List[str] = Field(
        ...,
        description="Correct list of Entities found absent from the summary that should be included in the next summary attempt.",
    )


# This multitask helper will be used to generate a chain of summaries.
# Allows us to extract data via streaming to see resuls faster
ChainOfDenseSummaries = instructor.MultiTask(
    Summary,
    name="chain-of-dense-summaries",
    description="""
        Repeat the following 2 steps 5 times.

            Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.

            Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

            A missing entity is:

            - relevant to the main story,
            - specific yet concise (5 words or fewer),
            - novel (not in the previous summary),
            - faithful (present in the article),
            - anywhere (can be located anywhere in the article).

            Remember, use the exact same number of words for each summary.""",
)


def summarize_article(article: str, n_summaries: int = 5, stream: bool = True):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        stream=stream,
        messages=[
            {
                "role": "system",
                "content": """Summarize the following article with {n_summary} chain of summaries with increasing density:""",
            },
            {"role": "user", "content": article},
        ],
        functions=[ChainOfDenseSummaries.openai_schema],
        function_call={"name": ChainOfDenseSummaries.openai_schema["name"]},
    )
    if stream:
        return ChainOfDenseSummaries.from_streaming_response(completion)
    return ChainOfDenseSummaries.from_response(completion)


if __name__ == "__main__":
    example = {
        "text": "The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.",
    }

    # Generate a chain of summaries, however we can also stream the results
    # to see the results faster
    for summary in summarize_article(example["text"]):
        pprint(summary.model_dump())

    """
    {'denser_summary': 'State agencies in California cannot enter into contracts '
                    'worth $100,000 or more with contractors that discriminate '
                    'in benefits based on gender identity. The requirement '
                    'applies to contractors operating within the state, on '
                    'state-owned or occupied property outside the state, and '
                    'elsewhere in the United States where work related to a '
                    'state contract is being performed. Contractors must treat '
                    'employee benefit requests and eligibility documentation as '
                    'confidential. Exceptions to the requirement can be made in '
                    'certain circumstances. Contractors can avoid being seen as '
                    'discriminatory if they pay the actual costs of benefits or '
                    'if they are unable to provide certain benefits despite '
                    'reasonable efforts. Contracts must include a certification '
                    'of compliance with the requirement.',
    'included_entities': ['California',
                        'contracts',
                        'discrimination',
                        'benefits',
                        'gender identity',
                        'state agencies',
                        'state-owned property',
                        'confidential',
                        'exceptions'],
    'index': 0,
    'missing_entities': []}
    {'denser_summary': 'State agencies in California cannot enter into contracts '
                    'worth $100,000 or more with contractors that discriminate '
                    'in benefits based on gender identity. The requirement '
                    'applies to contractors operating within the state, on '
                    'state-owned or occupied property outside the state, and '
                    'elsewhere in the United States where work related to a '
                    'state contract is being performed. Contractors must treat '
                    'employee benefit requests and eligibility documentation as '
                    'confidential. Exceptions to the requirement can be made in '
                    'certain circumstances, such as when there is only one '
                    'prospective contractor available or when the contract is '
                    'necessary to respond to an emergency. Contractors can '
                    'avoid being seen as discriminatory if they pay the actual '
                    'costs of benefits or if they are unable to provide certain '
                    'benefits despite reasonable efforts. Contracts must '
                    'include a certification of compliance with the '
                    'requirement, and false certification can result in '
                    'penalties.',
    'included_entities': ['California',
                        'contracts',
                        'discrimination',
                        'benefits',
                        'gender identity',
                        'state agencies',
                        'state-owned property',
                        'confidential',
                        'exceptions',
                        'prospective contractor',
                        'emergency',
                        'actual costs',
                        'penalties'],
    'index': 1,
    'missing_entities': ['availability', 'false certification']}
    {'denser_summary': 'State agencies in California are prohibited from entering '
                    'into contracts worth $100,000 or more with contractors '
                    'that discriminate in benefits based on gender identity. '
                    'This requirement applies to contractors operating within '
                    'the state, on state-owned or occupied property outside the '
                    'state, and elsewhere in the United States where work '
                    'related to a state contract is being performed. '
                    'Contractors must keep employee benefit requests and '
                    'eligibility documentation confidential. There are '
                    'exceptions to this requirement, such as when there is only '
                    'one available contractor or when an emergency situation '
                    'requires immediate contracting. Contractors can avoid '
                    'being seen as discriminatory by paying the actual costs of '
                    'benefits or if they are unable to provide certain benefits '
                    'despite reasonable efforts. Contracts must include a '
                    'certification of compliance with this requirement, and '
                    'false certification can lead to penalties and the '
                    'application of other existing remedies.',
    'included_entities': ['California',
                        'contracts',
                        'discrimination',
                        'benefits',
                        'gender identity',
                        'state agencies',
                        'state-owned property',
                        'confidential',
                        'exceptions',
                        'contractors',
                        'availability',
                        'emergency',
                        'actual costs',
                        'false certification',
                        'penalties'],
    'index': 2,
    'missing_entities': ['contracting practices', 'federal laws']}
    {'denser_summary': 'State agencies in California are prohibited from entering '
                    'into contracts worth $100,000 or more with contractors '
                    'that discriminate in benefits based on gender identity. '
                    'This requirement applies to contractors operating within '
                    'the state, on state-owned or occupied property outside the '
                    'state, and elsewhere in the United States where work '
                    'related to a state contract is being performed. '
                    'Contractors must keep employee benefit requests and '
                    'eligibility documentation confidential. There are '
                    'exceptions to this requirement, such as when there is only '
                    'one available contractor or when an emergency situation '
                    'requires immediate contracting. Contractors can avoid '
                    'being seen as discriminatory by paying the actual costs of '
                    'benefits or if they are unable to provide certain benefits '
                    'despite reasonable efforts. Contracts must include a '
                    'certification of compliance with this requirement, and '
                    'false certification can lead to penalties and the '
                    'application of other existing remedies. This section of '
                    'the Public Contract Code does not regulate the contracting '
                    'practices of local jurisdictions, and it is intended to be '
                    'consistent with applicable federal laws, rules, and '
                    'regulations.',
    'included_entities': ['California',
                        'contracts',
                        'discrimination',
                        'benefits',
                        'gender identity',
                        'state agencies',
                        'state-owned property',
                        'confidential',
                        'exceptions',
                        'contractors',
                        'availability',
                        'emergency',
                        'actual costs',
                        'false certification',
                        'penalties',
                        'Public Contract Code',
                        'local jurisdictions',
                        'federal laws',
                        'federal rules',
                        'federal regulations'],
    'index': 3,
    'missing_entities': []}
    """
