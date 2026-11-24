---
paths:
  - "report/**"
  - "docs/notes/**"
---
# Отчёт и тексты

Перед правкой или ревью прочитать целиком: `report/manuals/style.md`, `report/manuals/glossary.md`; для
`report/report.tex` — скилл `report`; для `report/example_model.ipynb` — ещё `notebooks/AGENTS.md`. Номера этапов
и найденные расхождения — `report/manuals/rewrite_plan.md`.

- Отчётные файлы `report/report.tex` и `report/example_model.ipynb` набираются по одному блоку (абзац или ячейка):
  текст блока — в ответе, в файл — после согласования (`AGENTS.md`, «Канон»).
- `report/report_tex/old_report.tex` — устаревший, только источник текста, не правится.
- `docs/notes/methods.md` правится только после подтверждения пользователя.
- Описание модели в отчёте по содержанию совпадает с её markdown-ячейкой; исходник — блокнот.
- Новое или изменённое обозначение и термин — в глоссарий в том же задании.
- `.tex` может быть сохранён редактором с CRLF — перед коммитом привести к LF.
