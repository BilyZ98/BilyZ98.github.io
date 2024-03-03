# Linux

## Lifetime setting for file
The `fcntl` function in Linux is used to manipulate file descriptors. It can perform various operations determined by the `cmd` argument¹. 

If you're looking to use read/write hints with `fcntl`, you might be referring to the `F_SET_RW_HINT` command. This command is used to set a hint for the expected read/write life-time of the file. 

Here's an example of how you might use it:

```c++
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <stdint.h>

static char const filename [] = "./file.txt";
static char const message [] = "/mnt/pool/my_file\\n";

int main (void) {
    int fp;
    int cnt = 0;
    errno = 0;
    uint64_t type = RWH_WRITE_LIFE_MEDIUM; // This is the write hint

    fp = open (filename, O_WRONLY | O_CREAT);
    if (fp == -1)
        return 1;

    fcntl (fp, F_SET_RW_HINT, &type); // Setting the write hint here

    cnt = write (fp, message, sizeof (message));
    if (cnt == -1)
        return 1;

    close (fp);
    return 0;
}

```

[https://www.man7.org/linux/man-pages/man2/fcntl.2.html](https://www.man7.org/linux/man-pages/man2/fcntl.2.html)

In this example, `RWH_WRITE_LIFE_MEDIUM` is the write hint, which is set using the `F_SET_RW_HINT` command in the `fcntl` function².


## Create filesystem across multiple disks

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
