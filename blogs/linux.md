To create a filesystem across multiple disks in Linux, you can use Logical Volume Management (LVM). Here are the steps:

1. **Identify the Disks**: Use the `fdisk -l` command to find the attached disks³.
2. **Create Partitions**: Use the `fdisk` command to create partitions on the disks³.
3. **Create Physical Volumes**: After partitioning the disks, use `pvcreate` to prepare your new partitions for LVM².
4. **Create a Volume Group**: Assign both physical volumes to a new volume group⁴.
5. **Create a Logical Volume**: Create a new logical volume in that volume group⁴.
6. **Create a Filesystem**: Decide the type of filesystem you want to create (such as ext4, XFS, etc.) and use the appropriate `mkfs` command variation. For example, for ext4, you can use `mkfs.ext4 /dev/sda1`¹.
7. **Mount the Filesystem**: After creating your filesystem, you can mount it in your operating system¹.
8. **Verify**: Finally, verify using `df -Th`³.

Please replace `/dev/sda1` with your actual device name. Be aware that these operations are potentially destructive and should be performed with care. Always ensure that your data is backed up before making such changes.

Here is an example of these commands:

```bash
# Create a physical volume on disk1 and disk2
pvcreate /dev/sda1 /dev/sdb1

# Create a volume group named myvg using the physical volumes
vgcreate myvg /dev/sda1 /dev/sdb1

# Create a logical volume named mylv in the volume group
lvcreate -l 100%FREE -n mylv myvg

# Create an ext4 filesystem in the logical volume
mkfs.ext4 /dev/myvg/mylv

# Mount the filesystem to a directory
mount /dev/myvg/mylv /mnt/mydir
```



- [Locks](./locks.md)
