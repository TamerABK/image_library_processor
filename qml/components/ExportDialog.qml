pragma ComponentBehavior: Bound
import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../theme" as Room36Theme

Dialog {
    id: root
    objectName: "exportDialog"
    property var exporter: null
    parent: Overlay.overlay
    anchors.centerIn: parent
    modal: true
    title: "Export photos"
    width: Math.min(520, parent ? parent.width - 40 : 520)
    closePolicy: exporter && exporter.busy ? Popup.NoAutoClose
                                           : Popup.CloseOnEscape | Popup.CloseOnPressOutside
    standardButtons: Dialog.NoButton
    onOpened: methodChoice.currentIndex = root.exporter && root.exporter.method === "hardlink" ? 1 : 0
    Connections {
        target: root.exporter
        function onChanged() {
            if (root.visible && !root.exporter.prepared && root.exporter.destination.length === 0
                    && root.exporter.summary.length === 0 && root.exporter.error.length === 0)
                root.close()
        }
    }
    Room36Theme.Theme { id: theme }
    Room36Theme.Typography { id: typography }
    background: Rectangle {
        radius: 20
        color: theme.panelFill
        border.width: 1
        border.color: theme.separator
    }
    contentItem: ColumnLayout {
        spacing: 12
        Text {
            Layout.fillWidth: true
            text: !root.exporter ? "" : root.exporter.planInfo.mode === "all_people"
                ? root.exporter.planInfo.peopleCount + " people · "
                    + root.exporter.planInfo.uniquePhotos + " unique photos · "
                    + root.exporter.planInfo.placements + " output files"
                : root.exporter.planInfo.placements + " photo"
                    + (root.exporter.planInfo.placements === 1 ? "" : "s") + " selected"
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            Accessible.name: text
        }
        Text {
            Layout.fillWidth: true
            text: root.exporter ? "Unique source size: "
                + (root.exporter.planInfo.totalBytes / (1024 * 1024)).toFixed(1) + " MB" : ""
            color: theme.mutedText
        }
        RoundedComboBox {
            id: methodChoice
            objectName: "exportMethod"
            Layout.fillWidth: true
            model: ["Copy", "Hard link"]
            enabled: root.exporter !== null && !root.exporter.busy
            onCurrentIndexChanged: if (root.exporter)
                root.exporter.setMethod(currentIndex === 1 ? "hardlink" : "copy")
            Accessible.name: "Export method: " + currentText
        }
        Text {
            Layout.fillWidth: true
            text: methodChoice.currentIndex === 1
                ? "Hard links share the source file's data. Edits through either path affect both; deleting one path leaves the other. Same drive required."
                : "Copy creates independent files and preserves available file metadata."
            color: theme.mutedText
            wrapMode: Text.WordWrap
            font.pixelSize: typography.caption
        }
        RowLayout {
            Layout.fillWidth: true
            PillButton {
                objectName: "exportChooseDestination"
                text: "Choose destination"
                variant: "translucent"
                enabled: root.exporter !== null && !root.exporter.busy
                onClicked: root.exporter.chooseDestination()
                Accessible.name: "Choose export destination folder"
            }
            Text {
                Layout.fillWidth: true
                text: root.exporter && root.exporter.destination ? root.exporter.destination : "No destination selected"
                color: theme.mainBlue
                elide: Text.ElideMiddle
                Accessible.name: text
            }
        }
        ProgressBar {
            objectName: "exportProgress"
            Layout.fillWidth: true
            visible: root.exporter !== null && root.exporter.busy
            from: 0
            to: root.exporter ? Math.max(1, root.exporter.progressTotal) : 1
            value: root.exporter ? root.exporter.progressDone : 0
            Accessible.name: "Export progress " + value + " of " + to
        }
        Text {
            Layout.fillWidth: true
            visible: root.exporter !== null && root.exporter.busy
            text: root.exporter ? root.exporter.progressDone + " of " + root.exporter.progressTotal + " files" : ""
            color: theme.mainBlue
        }
        Text {
            objectName: "exportSummary"
            Layout.fillWidth: true
            visible: root.exporter !== null && root.exporter.summary.length > 0
            text: root.exporter ? root.exporter.summary : ""
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            Accessible.name: text
        }
        Text {
            objectName: "exportError"
            Layout.fillWidth: true
            visible: root.exporter !== null && root.exporter.error.length > 0
            text: root.exporter ? root.exporter.error : ""
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            Accessible.name: text
        }
        Text {
            Layout.fillWidth: true
            visible: root.exporter !== null && root.exporter.failures.length > 0
            text: root.exporter ? root.exporter.failures.join("\n") : ""
            color: theme.mainBlue
            wrapMode: Text.WordWrap
            Accessible.name: text
        }
        RowLayout {
            Layout.alignment: Qt.AlignRight
            PillButton {
                objectName: "exportCancel"
                variant: "translucent"
                text: root.exporter && root.exporter.busy ? "Cancel export" : "Close"
                onClicked: {
                    if (root.exporter && root.exporter.busy) root.exporter.cancel()
                    else root.close()
                }
                Accessible.name: text
            }
            PillButton {
                objectName: "exportConfirm"
                variant: "accent"
                text: "Export"
                enabled: root.exporter !== null && root.exporter.prepared
                    && !root.exporter.busy && root.exporter.destination.length > 0
                onClicked: root.exporter.execute()
                Accessible.name: "Confirm photo export"
            }
        }
    }
}
