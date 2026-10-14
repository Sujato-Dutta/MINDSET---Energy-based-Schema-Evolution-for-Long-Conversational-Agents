"""Lexically disjoint synthetic confirmation; same families, not domain OOD."""
def confirmation_fixtures():
    groups={
        'same_scope_negation':[
            ('I own a telescope.','I do not own a telescope.',True),
            ('I am employed as a nurse.','I am not employed as a nurse.',True),
            ('I enjoy swimming.','I do not enjoy swimming.',True),
            ('I have exactly four rabbits right now.','I have exactly five rabbits right now.',True)],
        'coexisting_attributes':[
            ('I enjoy hiking.','I enjoy playing chess.',False),
            ('I work as a nurse.','I coach a basketball team.',False),
            ('My height is 170 centimeters.','My rent is 900 dollars.',False),
            ('I teach chemistry.','I do not teach literature.',False)],
        'different_contexts':[
            ('I drink coffee on weekends.','I drink water on weekdays.',False),
            ('I exercise in a park on Saturdays.','I exercise in a pool on Sundays.',False),
            ('I live in Munich during winter.','I live in Lisbon during summer.',False),
            ('My budget for books is 40 dollars.','My budget for food is 300 dollars.',False)],
        'different_times':[
            ('I worked as an engineer in 2018.','I work as a librarian in 2024.',False),
            ('I had one reptile in 2018.','I have three reptiles in 2024.',False),
            ('My gym membership cost 30 dollars last year.','My gym membership costs 40 dollars this year.',False),
            ('I volunteered at a shelter before 2015.','I manage a bakery after 2015.',False)],
        'plans_not_current_state':[
            ('I want to buy a bicycle.','I currently walk to school.',False),
            ('I plan to email Mira.','I plan to invite two colleagues.',False),
            ('I hope to graduate.','I am currently enrolled in school.',False),
            ('I might adopt a cat next month.','I do not own a cat now.',False)],
        'different_holders_or_objects':[
            ('My father teaches mathematics.','My mother teaches history.',False),
            ('My indoor plant is green.','My outdoor plant is yellow.',False),
            ('I own a silver bicycle.','I own a black motorcycle.',False),
            ('My parrot is five years old.','My fish is one year old.',False)]}
    return [{'id':f'confirmation:{family}:{i}','family':family,'left':a,'right':b,'expected_contradiction':label}
            for family,pairs in groups.items() for i,(a,b,label) in enumerate(pairs)]
