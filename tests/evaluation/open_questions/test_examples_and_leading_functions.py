"""Tests for examples and leading question functions."""

EXAMPLES_IN_QUESTIONS = [
    "What is your employer's main activity: teaching or research?",
    "Are you a student or a worker?",
    "Is your organisation mainly public or private?",
    "What is your employer's main activity, "
    "for example, providing finance, retail or social services?",
    "What is your employer's main activity, such as, "
    "hair cutting or teeth cleaning?",
    "Do you consider your job be in retail? E.g. selling goods in a shop.",
    "Do you see yourself as teenager or adult? I.e. "
    "13-19 years old or 20+ years old.",
    "Are you a student or a worker, including a lecturer as a worker?"
    "When did you decide to be a student, like a university student?",
]

EXPLICIT_EXAMPLE_MARKER_QUESTIONS = [
    "What is your employer's main activity, for example, "
    "providing finance, retail or social services?",
    "What products does your company make, for example, furniture or toys?",
    "What services does your organisation provide, for example, teaching or training?",
    "Do you consider your job to be in retail? E.g. selling goods in a shop.",
    "Do you work in manufacturing? E.g. making furniture or clothing.",
    "Are you self-employed? I.e. you run your own business.",
    "Do you work full time? I.e. 35 or more hours per week.",
    "What is your employer's main activity, such as hair cutting or teeth cleaning?",
    "What services do you provide, such as accounting or bookkeeping?",
    "Do you work in healthcare, like a nurse or doctor?",
    "Do you work in education, like a teacher or lecturer?",
]

INCLUDING_EXAMPLE_PHRASE_QUESTIONS = [
    "Are you a student or a worker, including a lecturer as a worker?",
    "Do you work in healthcare, including nursing roles?",
    "Does your role involve administration, including scheduling meetings?",
    "Does your role include customer service activities?",
    "Does your work include teaching apprentices?",
    "Does your job include managing staff?",
    "Do you work in manufacturing, for instance, producing furniture?",
    "What services do you provide, for instance, accounting or payroll support?",
    "What products does your employer make, for instance, bicycles or clothing?",
]

CATEGORY_OPTION_QUESTIONS = [
    "What is your employer's main activity: teaching or research?",
    "Are you employed in retail or manufacturing?",
    "Do you mainly provide products or services?",
    "Are you a manager or a supervisor?",
    "Do you work in the public or private sector?",
    "Are you involved in teaching or administration?",
    "What type of organisation do you work for: school, hospital or university?",
    "What kind of teacher are you: primary, secondary or college?",
]

PARENTHETICAL_EXAMPLE_QUESTIONS = [
    "What products does your employer manufacture (e.g. bicycles)?",
    "What service does your organisation provide (for example, hairdressing)?",
    "What type of work do you do (such as bookkeeping)?",
    "What industry do you work in (e.g. retail)?",
    "What services do you provide (for instance, accounting)?",
    "What products do you make (such as furniture)?",
]

EXAMPLE_LIST_AFTER_PUNCTUATION_QUESTIONS = [
    "What products does your employer manufacture, such as bicycles, scooters and motorcycles?",
    "What services do you provide, such as teaching, training and coaching?",
    "What roles do you recruit for, such as managers, analysts and developers?",
    "What products does your employer make - furniture, clothing or toys?",
    "What type of organisation do you work for - school, hospital or university?",
    "What products do you manufacture: furniture, bicycles and toys?",
]

DEFINITION_EXAMPLE_WORDING_QUESTIONS = [
    "Do you work in retail, meaning you sell goods directly to customers?",
    "Are you self-employed, meaning you run your own business?",
    "Do you work in education, meaning you teach or train people?",
    "Are you self-employed, which means you run your own business?",
    "Do you work in retail, which means selling goods to customers?",
    "Do you work in healthcare, namely nursing or physiotherapy?",
    "Do you provide professional services, namely accounting or legal advice?",
    "Do you work in education, that is teaching or training?",
    "Do you work in healthcare, that is nursing or physiotherapy?",
]
