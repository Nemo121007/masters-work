---
paths:
  - "notebooks/**/*.ipynb"
  - "notebooks/checks/**"
---
# Блокноты

Перед правкой или ревью прочитать целиком: `notebooks/AGENTS.md`; для markdown-ячеек — `report/style.md`
и `report/glossary.md`. Контекст текущей ревизии — `report/notebook_review.md`, решения — «Принятые решения»
в `report/rewrite_plan.md`.

- Большие блокноты целиком не читать: `grep -n`, затем `sed -n` по номерам строк.
- После правки — `python -m json.tool <файл> > /dev/null`.
- Рабочая копия блокнота пользователя (выводы, путь к данным, `SHOW_PLOTS`) сохраняется: в коммит — только
  правки задачи.
- Изменение кода модели — обновить её markdown; блокнот целиком запускает пользователь.
