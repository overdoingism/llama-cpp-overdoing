# stamps

Timestamps of commits of `main`, one directory per commit: `manifest.txt` lists the sha256 of the
commit's `git archive` and of each of its files; `freetsa.tsr` and `digicert.tsr` are RFC 3161
timestamps of the manifest (`request.tsq` is the query), `manifest.txt.ots` an OpenTimestamps proof.
Verify: `openssl ts -verify -in freetsa.tsr -data manifest.txt -CAfile <FreeTSA cacert.pem> -untrusted
<FreeTSA tsa.crt>`; `ots verify manifest.txt.ots`.  Written with Claude (AI).
