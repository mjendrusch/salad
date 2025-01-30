rsync --recursive --links --perms --times --compress --info=progress2 --port=33444 \
  rsync.wwpdb.org::ftp/data/assemblies/mmCIF/divided/ \
  "assembly_cif"
