### Read: TC1 amendment 49's re-ask -- UNTESTED again; the census places the field recipe's delta calls at 9,040 routed rows at most

- `tc1-5090-106` ($1.25, Xeon 8347C, quiet): the single block's draws unstable again (matched 13.4 %, shipped 8.0 % apart), the bucketed
  draws stable: P119-P122 UNTESTED. `TC1_PAD_CENSUS=1` measured every delta call: at most 9,040 routed rows at the field recipe (median 3,968),
  single blocks up to about 104,000 rows. Packed rows carry exactly 32,768 per call, so a routed-rows gate at 16,384 separates the two.
