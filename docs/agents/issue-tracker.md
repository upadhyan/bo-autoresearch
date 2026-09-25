# Issue tracker

GitHub Issues on `upadhyan/bo-autoresearch`, driven with the `gh` CLI.

## Wayfinding operations

- **Map**: an issue labelled `wayfinder:map`.
- **Ticket**: a native sub-issue of the map, labelled `wayfinder:<research|prototype|grilling|task>`.
  Add one with `gh api -X POST repos/upadhyan/bo-autoresearch/issues/<map>/sub_issues -F sub_issue_id=<ticket issue id>`. The id is the REST `id`, not the issue number.
- **Blocking**: native issue dependencies.
  `gh api -X POST repos/upadhyan/bo-autoresearch/issues/<blocked>/dependencies/blocked_by -F issue_id=<blocker issue id>`
- **Claim**: `gh issue edit <n> --add-assignee @me`. Do this before any work.
- **Frontier**: the map's open sub-issues that have no assignee and no open blockers.
  `gh api repos/upadhyan/bo-autoresearch/issues/<map>/sub_issues --paginate`, then for each open, unassigned one check `gh api repos/upadhyan/bo-autoresearch/issues/<n>/dependencies/blocked_by` for open blockers.
- **Resolve**: post a resolution comment, run `gh issue close <n>`, then append a line to the map's "Decisions so far".
