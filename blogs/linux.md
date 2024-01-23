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

Remember to replace `/dev/sda1`, `/dev/sdb1`, `myvg`, `mylv`, and `/mnt/mydir` with your actual device names, volume group name, logical volume name, and mount point respectively.

Source: Conversation with Bing, 1/23/2024
(1) Add multiple disks to Linux (real fast) | Since2k7. https://www.since2k7.com/blog/2018/11/29/add-multiple-disks-to-linux-real-fast/.
(2) How to merge multiple hard drives? - Unix & Linux Stack Exchange. https://unix.stackexchange.com/questions/329790/how-to-merge-multiple-hard-drives.
(3) How can I create one logical volume over two disks using LVM?. https://askubuntu.com/questions/219881/how-can-i-create-one-logical-volume-over-two-disks-using-lvm.
(4) How to create a filesystem on a Linux partition or logical volume. https://opensource.com/article/19/4/create-filesystem-linux-partition.
(5) How to combine multiple HDDs into one big HDD in Linux?. https://unix.stackexchange.com/questions/158336/how-to-combine-multiple-hdds-into-one-big-hdd-in-linux.
(6) Mounting multiple devices at a single mount point on Linux. https://unix.stackexchange.com/questions/32852/mounting-multiple-devices-at-a-single-mount-point-on-linux.
(7) undefined. https://www.howtoforge.com/linux_lvm.
