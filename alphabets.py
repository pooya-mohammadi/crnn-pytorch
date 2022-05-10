FA_LPR = dict(FA_ALPHABET='ابپتشثجدزسصطعفقکگلمنوهی+',
              FA_DIGITS='۰۱۲۳۴۵۶۷۸۹')

FA_DOCS = dict(FA_ALPHABET='اآبپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیى',
               FA_EXTRAS='ءؤئأّ',
               FA_DIGITS='۰۱۲۳۴۵۶۷۸۹',
               SYMBOLS='|()+-:،!.؛=%؟',
               EN_DIGITS='0123456789')

ALPHABETS = dict(FA_LPR=FA_LPR, FA_DOCS=FA_DOCS)


ALPHABETS = {k: "".join(list(v.values())) for k, v in ALPHABETS.items()}
