# ruword2tags - русский грамматический словарь - лексикон с морфологическими тегами

В составе пакета есть 2 класса: RuWord2Tags и RuFlexer, в которых сосредоточены
базовые операций с русской морфологией. RuWord2Tags разрабатывается как поставщик
грамматических признаков слов (фич) для алгоритмов машинного обучения. RuFlexer
содержит функционал для генеративных языковых моделей.

Для использования любого из классов нужно создать его экземпляр и затем вызвать у объекта
метод load(), чтобы загрузилась словарная база.


## Класс RuWord2Tags

Данный класс реализует единственную операцию - для заданного слова выдывать
все варианты наборов морфологического анализа в виде списков тегов. К примеру,
для слова *кошки* с помощью этого словаря можно определить, что это существительное
либо в форме именительного падежа множественного числа, либо в форме родительного
падежа единственного числа.

Для основных частей речи (существительные, прилагательные, глаголы) словарь содержит несколько
миллионов слов, обеспечивая распознавание большинства лексики в текстах. Кроме того,
для многих новых (out-of-vocabulary) слов пакет обеспечивает распознавание, даже если точно такого
слова нет в словаре.

Морфологическая информация при сборке пакета берется из текущей версии [Русского Грамматического Словаря](https://github.com/Koziev/GrammarEngine),
содержащего грамматическую информацию о более чем 4,3 миллионов словарных форм.

Для многих слов в русском языке набор грамматических тегов определяется неоднозначно, как в вышеупомянутом
примере с "кошки". В контексте предложения обычно можно снять эту неоднозначность, учитывая правила согласования
и синтаксис русского языка. Используйте [part-of-speech tagger](https://github.com/Koziev/rupostagger)
или [парсер](https://github.com/Koziev/GrammarEngine/tree/master/src/demo/ai/solarix/argon/ParseText/Parser) для такой работы.


## Класс RuFlexer

Основаная функция этого класса - подбор форм слов по лемме и набору грамматических признаков. Данная операция нужна
для построения генеративных моделей, например для аугментации NLP датасетов. К примеру, для леммы "кошка" и
набора тегов {падеж=твор, число=мн} мы получим "кошками". Далее есть пример вызова соответствующего метода.

## Совместимость

Пакет работает в питоне вер.3 под Windows и Linux, не требуя каких-либо внешних зависимостей.

## Установка

```
pip3 install git+https://github.com/Koziev/ruword2tags
```

## API

В текущей версии пакет требует, чтобы распознаваемые слова были заранее приведены
к нижнему регистру.

Словарь представлен экземпляром класса ruword2tags.RuWord2Tags. Словарь нужно
загрузить с диска перед использованием вызовом load().

Для распознавания одного слова нужно вызвать индексатор - см. пример далее.


## Примеры использования RuWord2Tags

```
import ruword2tags


# Создаем экземпляр словаря и загружаем его
word2tags = ruword2tags.RuWord2Tags()
word2tags.load()

# Получим теги для нескольких слов
for word in u'кошки рой для'.split():
	for i, tagset in enumerate(word2tags[word]):
		print(u'{}[{}] => {}'.format(word, i, tagset))
```

Результат:

```
кошки[0] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ИМ РОД=ЖЕН ЧИСЛО=МН
кошки[1] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=РОД РОД=ЖЕН ЧИСЛО=ЕД
рой[0] => ГЛАГОЛ ВИД=НЕСОВЕРШ ЛИЦО=2 НАКЛОНЕНИЕ=ПОБУД ПАДЕЖ=ВИН ПАДЕЖ=ДАТ ПАДЕЖ=ТВОР ЧИСЛО=ЕД
рой[1] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ВИН РОД=МУЖ ЧИСЛО=ЕД
рой[2] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ИМ РОД=МУЖ ЧИСЛО=ЕД
для[0] => ДЕЕПРИЧАСТИЕ ВИД=НЕСОВЕРШ ПАДЕЖ=ВИН
для[1] => ПРЕДЛОГ ПАДЕЖ=РОД
```

Возвращаемая индексатором класса ruword2tags.RuWord2Tags строка содержит набор тегов,
разделенных пробелом. Первый элемент - наименование части речи, далее идут теги
в формате ТЕГ=ЗНАЧЕНИЕ.

Для несловарных данных распознавание будет давать множество вариантов. Например:

```
for word in u'ккошки'.split():
	for i, tagset in enumerate(word2tags[word]):
		print(u'{}[{}] => {}'.format(word, i, tagset))
```

дает результаты:

```
ккошки[0] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ВИН РОД=СР ЧИСЛО=МН
ккошки[1] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=РОД РОД=МУЖ ЧИСЛО=ЕД
ккошки[2] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ИМ РОД=МУЖ ЧИСЛО=МН
ккошки[3] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ВИН РОД=ЖЕН ЧИСЛО=МН
ккошки[4] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ИМ РОД=ЖЕН ЧИСЛО=МН
ккошки[5] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=РОД РОД=ЖЕН ЧИСЛО=ЕД
ккошки[6] => СУЩЕСТВИТЕЛЬНОЕ ПАДЕЖ=ИМ РОД=СР ЧИСЛО=МН
```


## Пример использования RuFlexer

```
import ruword2tags

flexer = ruword2tags.RuFlexer()
flexer.load()

forms = flexer.find_forms_by_tags(u'кошка', [(u'ПАДЕЖ', u'ТВОР'), (u'ЧИСЛО', u'МН')])
print(list(forms))
```

дает результат:

```
['кошками']
```
