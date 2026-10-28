"""Spoken forms for numbers in Chinese and English text, using Python rules."""

import re
import unicodedata


DIGITS = '零一二三四五六七八九'
ONES = ('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen').split()
TENS = ('zero ten twenty thirty forty fifty sixty seventy eighty ninety').split()
NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
MONTHS = ('january february march april may june july august september october november december').split()


def chinese_integer(value):
    if len(value) > 1 and value[0] == '0':
        return ''.join(DIGITS[int(char)] for char in value)
    number = int(value)
    if number < 10:
        return DIGITS[number]
    for base, unit in ((10**12, '兆'), (10**8, '亿'), (10**4, '万')):
        if number >= base:
            high, low = divmod(number, base)
            return chinese_integer(str(high)) + unit + ('零' if 0 < low < base // 10 else '') + (chinese_integer(str(low)) if low else '')
    result, pending_zero = '', False
    for base, unit in ((1000, '千'), (100, '百'), (10, '十'), (1, '')):
        digit, number = divmod(number, base)
        if digit:
            result += ('零' if pending_zero else '') + DIGITS[digit] + unit
            pending_zero = False
        elif result and number:
            pending_zero = True
    return result[1:] if result.startswith('一十') else result


def english_integer(value):
    number = int(value)
    if number < 20:
        return ONES[number]
    if number < 100:
        high, low = divmod(number, 10)
        return TENS[high] + (' ' + ONES[low] if low else '')
    if number < 1000:
        high, low = divmod(number, 100)
        return ONES[high] + ' hundred' + (' and ' + english_integer(str(low)) if low else '')
    for base, unit in ((10**12, 'trillion'), (10**9, 'billion'), (10**6, 'million'), (1000, 'thousand')):
        if number >= base:
            high, low = divmod(number, base)
            return english_integer(str(high)) + ' ' + unit + (' ' + english_integer(str(low)) if low else '')


def spoken_number(value, chinese):
    value = value.replace(',', '')
    sign = ''
    if value[0] in '+-':
        sign = ('负' if value[0] == '-' else '正') if chinese else ('minus ' if value[0] == '-' else 'plus ')
        value = value[1:]
    whole, dot, fraction = value.partition('.')
    result = chinese_integer(whole) if chinese else english_integer(whole)
    if dot:
        result += '点' + ''.join(DIGITS[int(char)] for char in fraction) if chinese else ' point ' + ' '.join(ONES[int(char)] for char in fraction)
    return sign + result


def ordinal(value):
    words = english_integer(value).split()
    irregular = {'one':'first', 'two':'second', 'three':'third', 'five':'fifth', 'eight':'eighth', 'nine':'ninth', 'twelve':'twelfth'}
    last = words[-1]
    words[-1] = irregular.get(last, last[:-1] + 'ieth' if last.endswith('y') else last + 'th')
    return ' '.join(words)


def english_year(value):
    year = int(value)
    if 2010 <= year <= 2099 or 1100 <= year <= 1999:
        high, low = divmod(year, 100)
        return english_integer(str(high)) + (' ' + english_integer(str(low)) if low else ' hundred')
    return english_integer(value)


def normalize_numbers(text, chinese):
    text = unicodedata.normalize('NFKC', text)
    number = lambda value: spoken_number(value, chinese)
    digits = lambda value: ''.join(DIGITS[int(char)] for char in value) if chinese else ' '.join(ONES[int(char)] for char in value)
    operators = {'+':('加','plus'), '=':('等于','equals'), '×':('乘','times'), '÷':('除以','divided by')}
    text = re.sub(r'(?<=\d)\s*([+=×÷])\s*(?=[+-]?\d)',
                  lambda m: operators[m[1]][0] if chinese else ' ' + operators[m[1]][1] + ' ', text)

    def date(match):
        year, month, day = match.group(1, 2, 3)
        if chinese:
            return digits(year) + '年' + number(month) + '月' + number(day) + '日'
        month_index = int(month)
        if 1 <= month_index <= 12:
            return 'the ' + ordinal(day) + ' of ' + MONTHS[month_index - 1] + ', ' + english_year(year)
        return number(year) + ' slash ' + number(month) + ' slash ' + number(day)

    text = re.sub(r'(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)', date, text)
    if chinese:
        text = re.sub(r'(\d{4})年', lambda m: digits(m[1]) + '年', text)
    else:
        months_pattern = '|'.join(MONTHS)
        text = re.sub(r'\b(' + months_pattern + r')\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b',
                      lambda m: 'the ' + ordinal(m[2]) + ' of ' + m[1].lower() + ', ' + english_year(m[3]), text, flags=re.I)

    def clock(match):
        hour, minute, second, period = match.group(1, 2, 3, 4)
        if chinese:
            return number(hour) + '点' + (number(minute) + '分' if int(minute) else '') + (number(second) + '秒' if second else '') + (period or '')
        suffix = (' ' + period) if period else ''
        return number(hour) + ((' oh ' if minute[0] == '0' else ' ') + number(str(int(minute))) if int(minute) else '') + ((' ' + number(second) + ' seconds') if second else '') + suffix

    text = re.sub(r'(?<!\d)(\d{1,2}):(\d{2})(?::(\d{2}))?(?:\s*(AM|PM|am|pm))?', clock, text)

    def phone(match):
        groups = re.split(r'[- ]', match[0])
        return ''.join(digits(group) for group in groups) if chinese else ', '.join(digits(group) for group in groups)

    text = re.sub(r'(?<!\d)(?:\d{3,4}[- ]\d{3,4}[- ]\d{4}|1[3-9]\d{9})(?!\d)', phone, text)
    text = re.sub(r'(' + NUMBER + r')[eE]([+-]?\d+)',
                  lambda m: number(m[1]) + '乘十的' + number(m[2]) + '次方' if chinese else
                  number(m[1]) + ' times ten to the power of ' + number(m[2]), text)
    text = re.sub(r'(' + NUMBER + r')\s*%', lambda m: '百分之' + number(m[1]) if chinese else number(m[1]) + ' percent', text)
    text = re.sub(r'(' + NUMBER + r')\s*‰', lambda m: '千分之' + number(m[1]) if chinese else number(m[1]) + ' per mille', text)

    def money(match):
        symbol, value = match.group(1, 2)
        if chinese:
            return number(value) + {'¥':'元', '￥':'元', '$':'美元', '€':'欧元', '£':'英镑'}[symbol]
        if '.' in value:
            value = value.rstrip('0').rstrip('.')
        singular, plural = {'¥':('yuan','yuan'), '￥':('yuan','yuan'), '$':('dollar','dollars'), '€':('euro','euros'), '£':('pound','pounds')}[symbol]
        return number(value) + ' ' + (singular if value == '1' else plural)

    text = re.sub(r'([$¥￥€£])\s*(' + NUMBER + ')', money, text)
    if chinese:
        text = re.sub(r'(' + NUMBER + r')\s*(?:km/h|km/hour)', lambda m: '每小时' + number(m[1]) + '公里', text, flags=re.I)
        text = re.sub(r'(' + NUMBER + r')\s*m/s', lambda m: '每秒' + number(m[1]) + '米', text, flags=re.I)
    units = {
        'km/h':('公里每小时','kilometer per hour','kilometers per hour'), 'm/s':('米每秒','meter per second','meters per second'),
        '°C':('摄氏度','degree Celsius','degrees Celsius'), '°F':('华氏度','degree Fahrenheit','degrees Fahrenheit'),
        'km':('公里','kilometer','kilometers'), 'cm':('厘米','centimeter','centimeters'), 'mm':('毫米','millimeter','millimeters'),
        'kg':('千克','kilogram','kilograms'), 'mg':('毫克','milligram','milligrams'), 'g':('克','gram','grams'),
        'mL':('毫升','milliliter','milliliters'), 'ml':('毫升','milliliter','milliliters'), 'L':('升','liter','liters'),
        'm':('米','meter','meters'), 'kWh':('千瓦时','kilowatt hour','kilowatt hours'), 'kW':('千瓦','kilowatt','kilowatts'),
        'GB':('吉字节','gigabyte','gigabytes'), 'MB':('兆字节','megabyte','megabytes'), 'Hz':('赫兹','hertz','hertz'),
    }
    unit_pattern = '|'.join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
    def measurement(match):
        value, unit = match.group(1, 2)
        names = units[unit]
        return number(value) + names[0] if chinese else number(value) + ' ' + names[1 if value == '1' else 2]
    text = re.sub(r'(' + NUMBER + r')\s*(' + unit_pattern + r')(?![A-Za-z])', measurement, text)
    if chinese:
        text = re.sub(r'(?<!\d)(\d+)/(\d+)(?!\d)', lambda m: number(m[2]) + '分之' + number(m[1]), text)
    else:
        text = re.sub(r'(?<!\d)(\d+)(?:st|nd|rd|th)\b', lambda m: ordinal(m[1]), text)
        text = re.sub(r'(?<!\d)(\d+)/(\d+)(?!\d)', lambda m: number(m[1]) + ' over ' + number(m[2]), text)
    def remaining_number(match):
        result = number(match[0])
        if not chinese:
            if match.start() and text[match.start() - 1].isalnum():
                result = ' ' + result
            if match.end() < len(text) and text[match.end()].isalpha():
                result += ' '
        return result
    return re.sub(NUMBER, remaining_number, text)
